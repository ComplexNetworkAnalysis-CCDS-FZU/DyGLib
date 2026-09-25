import argparse
import sys
from typing import List, Literal, Optional
import torch
from pydantic.v1 import BaseModel, Field

from models.DyGFormer import DyGFormer
from models.SignDyGFormer import SignDyGFormer
from models.DirectSignDyGFormer import DirectSignDyGFormer
from models.NeighborInteractEncoder import TimeDecayGapMode
from utils.noise import NoiseScope
from utils import accel

EarlyStopLiteral = Literal[
    "exist_recall",
    "sign_f1",
    "ap",
    "f1_macro",
    "f1_binary",
    "acc",
    "auc",
    "f1_wt",
    "f1_mic",
    "f1_mac",
    "f1_bin",
    "f1_weighted",
    "f1_binary",
]


class SignPredictArgs(BaseModel):
    dataset_name: Literal[
        "WikiVote",
        "BitcoinAlpha",
        "BitcoinOTC",
        "RedditHyperlinkTitle",
        "RedditHyperlinkBody",
    ] = Field("BitcoinAlpha", description="dataset to be used")

    batch_size: int = Field(200, description="batch size")
    model_name: Literal[
        "DyGFormer",
        "SignDyGFormer",
        "DirectSignDyGFormer",
        "TGAT",
        "GraphMixer",
    ] = Field("SignDyGFormer", description="name of the model")

    gpu: int = Field(0, description="number of gpu to use")

    # 采样策略
    num_neighbors: int = Field(20, description="每个节点采样的邻居数量")
    sample_neighbor_strategy: Literal["uniform", "recent", "time_interval_aware"] = (
        Field("recent", description="历史邻居采样策略")
    )
    time_scaling_factor: float = Field(
        1e-6,
        description="控制时间间隔对采样的影响，更大的值倾向于采样更近的节点，当0时就是平均采样",
    )
    common_neighbors_look_forward: int = Field(
        1, description="每个共同邻居前向采样数量"
    )
    # 2026-09-15 双半径（k_c/k_r）：R 锚点（RAS 重复位置）的 look-forward 半径。
    # None = 与 common_neighbors_look_forward 相同（单一窗口，向后兼容、零行为变更）。
    ras_look_forward: Optional[int] = Field(
        None,
        description="RAS 重复锚点的 look-forward 半径 k_r；None=与 k_c 相同（原公式）",
    )
    pos_weight: float = Field(1.0, description="符号分类任务中的pos权重")

    # 模型参数
    num_heads: int = Field(2, description="注意力层中头的数量")
    num_layers: int = Field(2, description="模型层数量")
    seeds: List[int] = Field([2026], description="随机种子列表")
    seed:int=0
    # 潜空间维度
    time_feat_dim: int = Field(100, description="时间编码的维度")
    position_feat_dim: int = Field(172, description="位置编码的维度")
    patch_size: int = Field(1, description="切片大小")
    channel_embedding_dim: int = Field(
        50, description="各个通道的嵌入维度（邻居共现等）"
    )

    # 模型训练参数
    learning_rate: float = Field(0.0001)
    dropout: float = Field(0.1)
    num_epochs: int = Field(100)
    # num_runs: int = Field(4)
    optimizer: Literal["SGD", "Adam", "RMSprop"] = Field("Adam")

    weight_decay: float = Field(
        1e-4, description="L2正则化系数（权重衰减），缓解过拟合用"
    )

    patience: int = Field(20, description="早停机制的耐心度")
    early_stop_notice: Optional[List[EarlyStopLiteral]] = Field(
        None, description="早停模块关注的指标，不提供即关注全部指标"
    )

    time_gap: int = Field(
        2000,
        description="time gap for neighbors to compute node features",
    )

    # 数据集划分
    val_ratio: float = Field(0.15, description="验证集比例")
    test_ratio: float = Field(0.15, description="测试集比例")
    test_interval_epochs: int = Field(5, description="每隔多少epoch运行一次测试集")

    negative_sample_strategy: Literal["random", "historical", "inductive"] = Field(
        "random", description="负连边采样策略"
    )

    save_model_name: str = Field("")

    tail_num: Optional[int] = Field(None, description="部分数据集比例")

    reject_support: bool = Field(False, description="符号预测模块是否需要支持拒绝")

    exist_thr: Optional[float] = Field(None, description="边存在阈值，不提供自适应计算")
    sign_thr: Optional[float] = Field(None, description="符号阈值，不提供将自适应计算")

    # ---- 固定阈值评测 / eval-only（2026-09-23 用户批准）----
    # test_thr：sign（binary）任务测试阈值覆盖（提供则不使用自适应阈值）；
    # eval_only：跳过训练、装载 checkpoint 直接测试（结果名加 .EVT）；
    # 提供任一固定阈值时结果名加 .FX（防覆盖正常训练结果）。
    test_thr: Optional[float] = Field(
        None, description="sign 任务测试阈值覆盖（提供则不使用自适应阈值）"
    )
    eval_only: bool = Field(
        False, description="eval-only：跳过训练、装载 checkpoint 直接测试（结果名加 .EVT）"
    )
    # eval-only 装载的 checkpoint 基名（缺省 = result_save_name；用于固定阈值重评时
    # 指向原训练 run 的 ckpt，而结果写到带 .EVT/.FX 的新名下，互不覆盖）。
    eval_ckpt_name: str = Field(
        "", description="eval-only 装载的 checkpoint 基名（缺省=result_save_name）"
    )
    # D3b/T15 逐样本转储（2026-09-24）：非空 = 目录；评测时写 {目录}/{result_save_name}.npz
    # （逐样本 exist/sign 概率与标签，供全零/非全零分组与门控红线自查）。
    dump_samples: str = Field(
        "", description="逐样本转储目录（空=关；诊断用，不入结果名）"
    )

    module_repeat_aware_sampler: bool = Field(False, description="启用重复感知邻居采样")

    module_repeat_aware_sign_encoder: bool = Field(
        False, description="交互对重复交互对符号影响也考虑"
    )

    module_balance_theory_encoder: bool = Field(True, description="平衡理论编码模块")

    # ---- CNE 探针（2026-09-17 用户批准；Paper 6c7a 四）----
    # 共同邻居编码（CNE，邻居共现特征通道）开关；默认 True = 零行为变更，
    # 关闭（--no-module-common-neighbor-encoder）= 通道特征置零（结果名加 .CNE-D）。
    module_common_neighbor_encoder: bool = Field(
        True, description="共同邻居编码（CNE，共现特征）通道；关闭=置零探针（.CNE-D）"
    )
    # ---- E1a 空白填补（2026-09-22 用户批准；导师方向①"把最近交互补回来"）----
    # 取消 CNAS last-CN 截断：窗口并集尾部延伸至查询前一日。
    # 默认 False = 零行为变更；启用时结果名带 .TF-E（代际标记）+ strict-past 护栏断言。
    cnas_tail_fill: bool = Field(
        False, description="CNAS 空白填补（取消 last-CN 截断；结果名加 .TF-E）"
    )
    # ---- E1c 双块窗口（2026-09-22 用户方向，已认可）----
    # 窗口 = 最近 m 个事件（recency 块）∪ CNAS+RAS 锚点窗并集（证据块）；
    # 模型上限 = NN + m（两块各自保底；m=N 即"双倍 NN"）。
    # m=0 = 零行为变更；m>0 时结果名加 .RK-{m}（代际标记）。
    recent_block: int = Field(
        0, description="E1c 最近块大小 m（0=关；结果名加 .RK-m）"
    )
    # ---- E2 自历史锚点（2026-09-25 Paper 立项、用户已批）----
    # 锚点集合新增 R_self = 每侧最近 k 个历史交互（平铺并入；与 E1c 同机制、参数更小）。
    # 注意：无共同邻居且无重复的边走全历史回退（已含最近段），E2 净增量落在有锚点边上。
    # k=0 = 零行为变更；k>0 时结果名加 .E2-{k}；模型上限 = NN + k。
    e2_self_recent: int = Field(
        0, description="E2 自历史锚点：每侧最近 k 个事件并入锚点集合（0=关；结果名加 .E2-k）"
    )
    # ---- G1 证据存在性门控（2026-09-22 用户批准；对抗"全零 BTE 有害"假设）----
    # True = 无证据位置（(pos,neg) 双零）的 BTE 分支输出置零；默认 False = 零行为变更；
    # 结果名加 .G1（代际标记）。
    module_bte_evidence_gate: bool = Field(
        False, description="G1 证据存在性门控（无证据位置 BTE 分支置零；结果名加 .G1）"
    )

    # ---- BTE 微调四变体（2026-09-24 Paper d350，用户已批）----
    # 统一：不新增模块、不动 BTE 公式；默认关 = 零行为变更；结果名加 .B2/.B3/.B4/.B5。
    # A 族（无证据位置处理）互斥：G1 / B5 / B3；B 族（有证据时缩放）：B2 / B4，独立旋钮。
    module_bte_b2_density_norm: bool = Field(
        False, description="B2 证据密度归一化（按样本非零证据位置数 sqrt 归一；无参；结果名加 .B2）"
    )
    module_bte_b3_default_marker: bool = Field(
        False, description="B3 缺省证据标记（无证据位置可学习常量，init=现行默认；+F 参数；结果名加 .B3）"
    )
    module_bte_b4_channel_gate: bool = Field(
        False, description="B4 符号分通道加权（pos/neg 分通道路径可学习增益，init 恒等；+2 参数；结果名加 .B4）"
    )
    module_bte_b5_continuous_gate: bool = Field(
        False, description="B5 连续门控（g=min(1,sqrt(n/τ))，τ=批内中位数；无参；结果名加 .B5）"
    )

    # ---- E-4: 时间衰减证据构造（lambda 为 None 表示不启用）----
    time_decay_lambda: Optional[float] = Field(
        None, description="时间衰减系数 λ；不提供(None)则不启用时间衰减"
    )
    time_decay_gap_mode: TimeDecayGapMode = Field(
        TimeDecayGapMode.STALENESS,
        description="Δt 定义: staleness=证据陈旧度(A), gap=事件间隔(B)",
    )

    module_status_theory_encoder: bool = Field(
        True, description="状态理论编码模块 (DirectSignDyGFormer)"
    )

    module_balance_fallback: bool = Field(
        True, description="状态理论不确定时退避到平衡理论 (DirectSignDyGFormer)"
    )

    # ---- E-7: 噪声鲁棒性（可插拔噪声模块，与 SEMBA 仓库共用 seed 保证公平）----
    noise_ratio: Optional[float] = Field(
        None, description="符号翻转噪声比例（0-1）；None 表示不加噪"
    )
    noise_seed: int = Field(0, description="噪声翻转随机种子（与 SEMBA 仓库一致）")
    noise_scope: NoiseScope = Field(
        NoiseScope.TRAIN, description="噪声施加范围: TRAIN=仅训练集加噪, ALL=全数据加噪"
    )

    module_common_neighbor_aware_sampler: bool = Field(
        True, description="历史共邻居采样感知模块"
    )

    # ---- BTE 门控（预留接口，默认禁用；启用后在 SignDyGFormer 第 5 通道上加 sigmoid 门）----
    module_balance_theory_gate: bool = Field(
        False, description="BTE 通道门控（预留接口，默认禁用；后续或有向图实验用）"
    )

    # ---- M4: 加速后端开关（命令行显式控制；默认启用）----
    accel: bool = Field(
        True,
        description="启用 signdyg_accel 加速后端（默认启用；--no-accel 关闭，--accel 亦可显式启用）",
    )

    ablation:bool =Field(False,description="消融实验模式")

    @property
    def device(self):
        return (
            f"cuda:{self.gpu}" if torch.cuda.is_available() and self.gpu >= 0 else "cpu"
        )

    @property
    def result_save_name(self):
        look_forward = self.common_neighbors_look_forward
        sample_number = self.num_neighbors

        if self.ablation:
            param_fmt = ".NN-Best.LF-Best"
        else:
            param_fmt = f".NN-{sample_number}.LF-{look_forward}"

        # 双半径标记：显式设置 k_r 时始终带上 .RLF-{k_r}（含 k_r==k_c 的对角点），
        # 与单窗口同配置结果在文件名上区分，避免覆盖，便于聚合脚本按标签过滤。
        if self.ras_look_forward is not None:
            param_fmt += f".RLF-{self.ras_look_forward}"

        enable_RAS = self.module_repeat_aware_sampler
        enable_RASE = self.module_repeat_aware_sign_encoder
        enable_BTE = self.module_balance_theory_encoder
        enable_CNAS = self.module_common_neighbor_aware_sampler

        bool2str = lambda x: "E" if x else "D"

        # E-4: 时间编码(TE) / 时间衰减(TD) 标记，避免两者结果覆盖
        td_tag = "TD" if self.time_decay_lambda is not None else "TE"

        # E-7: 噪声标记（比例+范围），避免与干净结果覆盖
        noise_tag = ""
        if self.noise_ratio is not None and self.noise_ratio > 0:
            noise_tag = (
                f".N{int(round(self.noise_ratio * 100))}"
                f"{'T' if self.noise_scope == NoiseScope.TRAIN else 'A'}"
            )

        # CNE 探针标记（2026-09-17）：通道关闭时追加 .CNE-D，避免与常开结果互相覆盖；
        # 默认开启 ⇒ 无标记，命名与历史结果完全一致（零行为变更）。
        cne_tag = "" if self.module_common_neighbor_encoder else ".CNE-D"

        # E1a 空白填补标记（2026-09-22）：启用时追加 .TF-E（代际标记；跨代不可比）
        tf_tag = ".TF-E" if self.cnas_tail_fill else ""

        # E1c 双块窗口标记（2026-09-22）：m>0 时追加 .RK-{m}（代际标记）
        rk_tag = f".RK-{self.recent_block}" if self.recent_block > 0 else ""
        e2_tag = f".E2-{self.e2_self_recent}" if self.e2_self_recent > 0 else ""

        # G1 证据存在性门控标记（2026-09-22）：启用时追加 .G1（代际标记）
        g1_tag = ".G1" if self.module_bte_evidence_gate else ""
        # BTE 微调四变体标记（2026-09-24）：各自追加 .B2/.B3/.B4/.B5
        b2_tag = ".B2" if self.module_bte_b2_density_norm else ""
        b3_tag = ".B3" if self.module_bte_b3_default_marker else ""
        b4_tag = ".B4" if self.module_bte_b4_channel_gate else ""
        b5_tag = ".B5" if self.module_bte_b5_continuous_gate else ""

        # 固定阈值 / eval-only 标记（2026-09-23 用户批准；防覆盖正常训练结果）
        evt_tag = ".EVT" if self.eval_only else ""
        fx_tag = (
            ".FX"
            if (
                self.test_thr is not None
                or self.sign_thr is not None
                or self.exist_thr is not None
            )
            else ""
        )

        # E-3: patch 标记，避免不同 patch-size 结果互相覆盖（2026-09-06 修复）
        # 注：始终带上 .P{size}，使主表/消融(.P1) 与 patch 扫描各组在文件名上互相区分；
        #     旧 CPU 期文件无 .P 标记，天然可辨认为历史数据。
        return (
            f"{self.save_model_name}{param_fmt}"
            f".RAS-{bool2str(enable_RAS)}.RASE-{bool2str(enable_RASE)}"
            f".BTE-{bool2str(enable_BTE)}.CNAS-{bool2str(enable_CNAS)}"
            f".P{self.patch_size}"
            f".{td_tag}{noise_tag}{cne_tag}{tf_tag}{rk_tag}{e2_tag}{g1_tag}{b2_tag}{b3_tag}{b4_tag}{b5_tag}{evt_tag}{fx_tag}"
            # BTE 门控标记（预留接口，默认禁用；启用时避免与无门控结果互相覆盖）
            + (".GATE" if self.module_balance_theory_gate else "")
        )
    @property
    def num_runs(self):
        return len(self.seeds)
    
    @property
    def max_input_sequence_length(self):
        # E1c 双块窗口：模型上限 = NN + m（证据块 N + 最近块 m；m=0 时为原值）
        # E2（2026-09-25）：上限再叠加 k（每侧最近 k 保底；两块都与 NN 互不挤）
        return (
            self.num_neighbors
            + max(0, self.recent_block)
            + max(0, self.e2_self_recent)
        )

def get_sign_prediction_args(is_evaluation=False):
    import pydantic_argparse

    # M4：兼容显式 `--accel`——pydantic_argparse 对默认-True 布尔只生成 `--no-accel`，
    # 这里把 `--accel` 视为“显式启用”（优先于环境变量）并从 argv 剥离，提前于解析执行。
    _explicit_accel = "--accel" in sys.argv
    if _explicit_accel:
        sys.argv = [arg for arg in sys.argv if arg != "--accel"]

    parser = pydantic_argparse.ArgumentParser(SignPredictArgs)

    try:
        args = parser.parse_typed_args()
    except:
        parser.print_help()
        sys.exit()

    # M4：加速开关（CLI 优先 → 环境变量 SIGNDYG_ACCEL 兜底 → 未设=默认启用；
    #     启动即严格校验，fail-fast）
    if not args.accel:
        accel.configure(False)
    elif _explicit_accel:
        accel.configure(True)
    else:
        accel.configure(None)

    return args


def get_link_prediction_args(is_evaluation: bool = False):
    """
    get the args for the link prediction task
    :param is_evaluation: boolean, whether in the evaluation process
    :return:
    """
    # arguments
    parser = argparse.ArgumentParser("Interface for the link prediction task")
    parser.add_argument(
        "--dataset_name",
        type=str,
        help="dataset to be used",
        default="wikipedia",
        choices=[
            "wikipedia",
            "reddit",
            "mooc",
            "lastfm",
            "myket",
            "enron",
            "SocialEvo",
            "uci",
            "Flights",
            "CanParl",
            "USLegis",
            "UNtrade",
            "UNvote",
            "Contacts",
            "WikiVote",
            "BitcoinAlpha",
            "BitcoinOTC",
            "RedditHyperlinkTitle",
            "RedditHyperlinkBody",
        ],
    )
    parser.add_argument("--batch_size", type=int, default=200, help="batch size")
    parser.add_argument(
        "--model_name",
        type=str,
        default=SignDyGFormer.NAME,
        help="name of the model, note that EdgeBank is only applicable for evaluation",
        choices=[
            "JODIE",
            "DyRep",
            "TGAT",
            "TGN",
            "CAWN",
            "EdgeBank",
            "TCL",
            "GraphMixer",
            DyGFormer.NAME,
            SignDyGFormer.NAME,
            DirectSignDyGFormer.NAME,
        ],
    )
    parser.add_argument("--gpu", type=int, default=0, help="number of gpu to use")
    parser.add_argument(
        "--num_neighbors",
        type=int,
        default=20,
        help="number of neighbors to sample for each node",
    )
    parser.add_argument(
        "--sample_neighbor_strategy",
        type=str,
        default="recent",
        choices=["uniform", "recent", "time_interval_aware"],
        help="how to sample historical neighbors",
    )
    parser.add_argument(
        "--time_scaling_factor",
        default=1e-6,
        type=float,
        help="the hyperparameter that controls the sampling preference with time interval, "
        "a large time_scaling_factor tends to sample more on recent links, 0.0 corresponds to uniform sampling, "
        "it works when sample_neighbor_strategy == time_interval_aware",
    )
    parser.add_argument(
        "--num_walk_heads",
        type=int,
        default=8,
        help="number of heads used for the attention in walk encoder",
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=2,
        help="number of heads used in attention layer",
    )
    parser.add_argument(
        "--num_layers", type=int, default=2, help="number of model layers"
    )
    parser.add_argument(
        "--walk_length", type=int, default=1, help="length of each random walk"
    )
    parser.add_argument(
        "--time_gap",
        type=int,
        default=2000,
        help="time gap for neighbors to compute node features",
    )
    parser.add_argument(
        "--time_feat_dim", type=int, default=100, help="dimension of the time embedding"
    )
    parser.add_argument(
        "--position_feat_dim",
        type=int,
        default=172,
        help="dimension of the position embedding",
    )
    parser.add_argument(
        "--edge_bank_memory_mode",
        type=str,
        default="unlimited_memory",
        help="how memory of EdgeBank works",
        choices=["unlimited_memory", "time_window_memory", "repeat_threshold_memory"],
    )
    parser.add_argument(
        "--time_window_mode",
        type=str,
        default="fixed_proportion",
        help="how to select the time window size for time window memory",
        choices=["fixed_proportion", "repeat_interval"],
    )
    parser.add_argument("--patch_size", type=int, default=1, help="patch size")
    parser.add_argument(
        "--channel_embedding_dim",
        type=int,
        default=50,
        help="dimension of each channel embedding",
    )
    parser.add_argument(
        "--max_input_sequence_length",
        type=int,
        default=32,
        help="maximal length of the input sequence of each node",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.0001, help="learning rate"
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout rate")
    parser.add_argument("--num_epochs", type=int, default=100, help="number of epochs")
    parser.add_argument(
        "--optimizer",
        type=str,
        default="Adam",
        choices=["SGD", "Adam", "RMSprop"],
        help="name of optimizer",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight decay")
    parser.add_argument(
        "--patience", type=int, default=20, help="patience for early stopping"
    )
    parser.add_argument(
        "--val_ratio", type=float, default=0.15, help="ratio of validation set"
    )
    parser.add_argument(
        "--test_ratio", type=float, default=0.15, help="ratio of test set"
    )
    parser.add_argument("--num_runs", type=int, default=5, help="number of runs")
    parser.add_argument(
        "--test_interval_epochs",
        type=int,
        default=10,
        help="how many epochs to perform testing once",
    )
    parser.add_argument(
        "--negative_sample_strategy",
        type=str,
        default="random",
        choices=["random", "historical", "inductive"],
        help="strategy for the negative edge sampling",
    )
    parser.add_argument(
        "--load_best_configs",
        action="store_true",
        default=False,
        help="whether to load the best configurations",
    )

    parser.add_argument(
        "--pos_weight", type=float, help="符号分类任务中的pos权重", default=1.0
    )

    try:
        args = parser.parse_args()
        args.device = (
            f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu"
        )
    except:
        parser.print_help()
        sys.exit()

    if args.model_name == "EdgeBank":
        assert is_evaluation, "EdgeBank is only applicable for evaluation!"

    if args.load_best_configs:
        load_link_prediction_best_configs(args=args)

    return args


def load_link_prediction_best_configs(args: argparse.Namespace):
    """
    load the best configurations for the link prediction task
    :param args: argparse.Namespace
    :return:
    """
    # model specific settings
    if args.model_name == "TGAT":
        args.num_neighbors = 20
        args.num_layers = 2
        if args.dataset_name in ["enron", "CanParl", "UNvote"]:
            args.dropout = 0.2
        else:
            args.dropout = 0.1
        if args.dataset_name in ["reddit", "CanParl", "UNtrade"]:
            args.sample_neighbor_strategy = "uniform"
        else:
            args.sample_neighbor_strategy = "recent"
    elif args.model_name in ["JODIE", "DyRep", "TGN"]:
        args.num_neighbors = 10
        args.num_layers = 1
        if args.model_name == "JODIE":
            if args.dataset_name in ["mooc", "USLegis"]:
                args.dropout = 0.2
            elif args.dataset_name in ["lastfm"]:
                args.dropout = 0.3
            elif args.dataset_name in ["uci", "UNtrade"]:
                args.dropout = 0.4
            elif args.dataset_name in ["CanParl"]:
                args.dropout = 0.0
            else:
                args.dropout = 0.1
        elif args.model_name == "DyRep":
            if args.dataset_name in [
                "mooc",
                "lastfm",
                "enron",
                "uci",
                "CanParl",
                "USLegis",
                "Contacts",
            ]:
                args.dropout = 0.0
            else:
                args.dropout = 0.1
        else:
            assert args.model_name == "TGN"
            if args.dataset_name in ["mooc", "UNtrade"]:
                args.dropout = 0.2
            elif args.dataset_name in ["lastfm", "CanParl"]:
                args.dropout = 0.3
            elif args.dataset_name in ["enron", "SocialEvo"]:
                args.dropout = 0.0
            else:
                args.dropout = 0.1
        if args.model_name in ["TGN", "DyRep"]:
            if args.dataset_name in ["CanParl"] or (
                args.model_name == "TGN" and args.dataset_name == "UNvote"
            ):
                args.sample_neighbor_strategy = "uniform"
            else:
                args.sample_neighbor_strategy = "recent"
    elif args.model_name == "CAWN":
        args.time_scaling_factor = 1e-6
        if args.dataset_name in [
            "mooc",
            "SocialEvo",
            "uci",
            "Flights",
            "UNtrade",
            "UNvote",
            "Contacts",
        ]:
            args.num_neighbors = 64
        elif args.dataset_name in ["lastfm", "CanParl"]:
            args.num_neighbors = 128
        else:
            args.num_neighbors = 32
        if args.dataset_name in ["CanParl"]:
            args.dropout = 0.0
        else:
            args.dropout = 0.1
        args.sample_neighbor_strategy = "time_interval_aware"
    elif args.model_name == "EdgeBank":
        if args.negative_sample_strategy == "random":
            if args.dataset_name in ["wikipedia", "reddit", "uci", "Flights"]:
                args.edge_bank_memory_mode = "unlimited_memory"
            elif args.dataset_name in ["mooc", "lastfm", "enron", "CanParl", "USLegis"]:
                args.edge_bank_memory_mode = "time_window_memory"
                args.time_window_mode = "fixed_proportion"
            elif args.dataset_name in ["UNtrade", "UNvote", "Contacts"]:
                args.edge_bank_memory_mode = "time_window_memory"
                args.time_window_mode = "repeat_interval"
            else:
                assert args.dataset_name == "SocialEvo"
                args.edge_bank_memory_mode = "repeat_threshold_memory"
        elif args.negative_sample_strategy == "historical":
            if args.dataset_name in ["uci", "CanParl", "USLegis"]:
                args.edge_bank_memory_mode = "time_window_memory"
                args.time_window_mode = "fixed_proportion"
            elif args.dataset_name in [
                "mooc",
                "lastfm",
                "enron",
                "UNtrade",
                "UNvote",
                "Contacts",
            ]:
                args.edge_bank_memory_mode = "time_window_memory"
                args.time_window_mode = "repeat_interval"
            else:
                assert args.dataset_name in [
                    "wikipedia",
                    "reddit",
                    "SocialEvo",
                    "Flights",
                ]
                args.edge_bank_memory_mode = "repeat_threshold_memory"
        else:
            assert args.negative_sample_strategy == "inductive"
            if args.dataset_name in ["USLegis"]:
                args.edge_bank_memory_mode = "time_window_memory"
                args.time_window_mode = "fixed_proportion"
            elif args.dataset_name in ["uci", "UNvote"]:
                args.edge_bank_memory_mode = "time_window_memory"
                args.time_window_mode = "repeat_interval"
            else:
                assert args.dataset_name in [
                    "wikipedia",
                    "reddit",
                    "mooc",
                    "lastfm",
                    "myket",
                    "enron",
                    "SocialEvo",
                    "Flights",
                    "CanParl",
                    "UNtrade",
                    "Contacts",
                ]
                args.edge_bank_memory_mode = "repeat_threshold_memory"
    elif args.model_name == "TCL":
        args.num_neighbors = 20
        args.num_layers = 2
        if args.dataset_name in ["SocialEvo", "uci", "UNtrade", "UNvote", "Contacts"]:
            args.dropout = 0.0
        elif args.dataset_name in ["CanParl"]:
            args.dropout = 0.2
        elif args.dataset_name in ["USLegis"]:
            args.dropout = 0.3
        else:
            args.dropout = 0.1
        if args.dataset_name in ["reddit", "CanParl", "USLegis", "UNtrade", "UNvote"]:
            args.sample_neighbor_strategy = "uniform"
        else:
            args.sample_neighbor_strategy = "recent"
    elif args.model_name == "GraphMixer":
        args.num_layers = 2
        if args.dataset_name in ["wikipedia"]:
            args.num_neighbors = 30
        elif args.dataset_name in ["reddit", "lastfm"]:
            args.num_neighbors = 10
        else:
            args.num_neighbors = 20
        if args.dataset_name in ["wikipedia", "reddit", "enron"]:
            args.dropout = 0.5
        elif args.dataset_name in ["mooc", "uci", "USLegis"]:
            args.dropout = 0.4
        elif args.dataset_name in ["lastfm", "UNvote"]:
            args.dropout = 0.0
        elif args.dataset_name in ["SocialEvo"]:
            args.dropout = 0.3
        elif args.dataset_name in ["Flights", "CanParl"]:
            args.dropout = 0.2
        else:
            args.dropout = 0.1
        if args.dataset_name in ["CanParl", "UNtrade", "UNvote"]:
            args.sample_neighbor_strategy = "uniform"
        else:
            args.sample_neighbor_strategy = "recent"
    elif args.model_name == SignDyGFormer.NAME:
        args.num_layers = 2
        if args.dataset_name in ["reddit"]:
            args.max_input_sequence_length = 64
            args.patch_size = 2
        elif args.dataset_name in ["mooc", "enron", "Flights", "USLegis", "UNtrade"]:
            args.max_input_sequence_length = 256
            args.patch_size = 8
        elif args.dataset_name in ["lastfm"]:
            args.max_input_sequence_length = 512
            args.patch_size = 16
        elif args.dataset_name in ["CanParl"]:
            args.max_input_sequence_length = 2048
            args.patch_size = 64
        elif args.dataset_name in ["UNvote"]:
            args.max_input_sequence_length = 128
            args.patch_size = 4
        else:
            args.max_input_sequence_length = 32
            args.patch_size = 1
        assert args.max_input_sequence_length % args.patch_size == 0
        if args.dataset_name in ["reddit", "UNvote"]:
            args.dropout = 0.2
        elif args.dataset_name in ["enron", "USLegis", "UNtrade", "Contacts"]:
            args.dropout = 0.0
        else:
            args.dropout = 0.1
    else:
        raise ValueError(f"Wrong value for model_name {args.model_name}!")


def get_node_classification_args():
    """
    get the args for the node classification task
    :return:
    """
    # arguments
    parser = argparse.ArgumentParser("Interface for the node classification task")
    parser.add_argument(
        "--dataset_name",
        type=str,
        help="dataset to be used",
        default="wikipedia",
        choices=["wikipedia", "reddit"],
    )
    parser.add_argument("--batch_size", type=int, default=200, help="batch size")
    parser.add_argument(
        "--model_name",
        type=str,
        default=DyGFormer.NAME,
        help="name of the model",
        choices=[
            "JODIE",
            "DyRep",
            "TGAT",
            "TGN",
            "CAWN",
            "TCL",
            "GraphMixer",
            DyGFormer.NAME,
        ],
    )
    parser.add_argument("--gpu", type=int, default=0, help="number of gpu to use")
    parser.add_argument(
        "--num_neighbors",
        type=int,
        default=20,
        help="number of neighbors to sample for each node",
    )
    parser.add_argument(
        "--sample_neighbor_strategy",
        type=str,
        default="recent",
        choices=["uniform", "recent", "time_interval_aware"],
        help="how to sample historical neighbors",
    )
    parser.add_argument(
        "--time_scaling_factor",
        default=1e-6,
        type=float,
        help="the hyperparameter that controls the sampling preference with time interval, "
        "a large time_scaling_factor tends to sample more on recent links, 0.0 corresponds to uniform sampling, "
        "it works when sample_neighbor_strategy == time_interval_aware",
    )
    parser.add_argument(
        "--num_walk_heads",
        type=int,
        default=8,
        help="number of heads used for the attention in walk encoder",
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=2,
        help="number of heads used in attention layer",
    )
    parser.add_argument(
        "--num_layers", type=int, default=2, help="number of model layers"
    )
    parser.add_argument(
        "--walk_length", type=int, default=1, help="length of each random walk"
    )
    parser.add_argument(
        "--time_gap",
        type=int,
        default=2000,
        help="time gap for neighbors to compute node features",
    )
    parser.add_argument(
        "--time_feat_dim", type=int, default=100, help="dimension of the time embedding"
    )
    parser.add_argument(
        "--position_feat_dim",
        type=int,
        default=172,
        help="dimension of the position embedding",
    )
    parser.add_argument("--patch_size", type=int, default=1, help="patch size")
    parser.add_argument(
        "--channel_embedding_dim",
        type=int,
        default=50,
        help="dimension of each channel embedding",
    )
    parser.add_argument(
        "--max_input_sequence_length",
        type=int,
        default=32,
        help="maximal length of the input sequence of each node",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.0001, help="learning rate"
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout rate")
    parser.add_argument("--num_epochs", type=int, default=100, help="number of epochs")
    parser.add_argument(
        "--optimizer",
        type=str,
        default="Adam",
        choices=["SGD", "Adam", "RMSprop"],
        help="name of optimizer",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight decay")
    parser.add_argument(
        "--patience", type=int, default=20, help="patience for early stopping"
    )
    parser.add_argument(
        "--val_ratio", type=float, default=0.15, help="ratio of validation set"
    )
    parser.add_argument(
        "--test_ratio", type=float, default=0.15, help="ratio of test set"
    )
    parser.add_argument("--num_runs", type=int, default=5, help="number of runs")
    parser.add_argument(
        "--test_interval_epochs",
        type=int,
        default=10,
        help="how many epochs to perform testing once",
    )
    parser.add_argument(
        "--load_best_configs",
        action="store_true",
        default=False,
        help="whether to load the best configurations",
    )

    try:
        args = parser.parse_args()
        args.device = (
            f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu"
        )
    except:
        parser.print_help()
        sys.exit()

    assert args.dataset_name in [
        "wikipedia",
        "reddit",
    ], f"Wrong value for dataset_name {args.dataset_name}!"
    if args.load_best_configs:
        load_node_classification_best_configs(args=args)

    return args


def load_node_classification_best_configs(args: argparse.Namespace):
    """
    load the best configurations for the node classification task
    :param args: argparse.Namespace
    :return:
    """
    # model specific settings
    if args.model_name == "TGAT":
        args.num_neighbors = 20
        args.num_layers = 2
        args.dropout = 0.1
        if args.dataset_name in ["reddit"]:
            args.sample_neighbor_strategy = "uniform"
        else:
            args.sample_neighbor_strategy = "recent"
    elif args.model_name in ["JODIE", "DyRep", "TGN"]:
        args.num_neighbors = 10
        args.num_layers = 1
        args.dropout = 0.1
        args.sample_neighbor_strategy = "recent"
    elif args.model_name == "CAWN":
        args.time_scaling_factor = 1e-6
        args.num_neighbors = 32
        args.dropout = 0.1
        args.sample_neighbor_strategy = "time_interval_aware"
    elif args.model_name == "TCL":
        args.num_neighbors = 20
        args.num_layers = 2
        args.dropout = 0.1
        if args.dataset_name in ["reddit"]:
            args.sample_neighbor_strategy = "uniform"
        else:
            args.sample_neighbor_strategy = "recent"
    elif args.model_name == "GraphMixer":
        args.num_layers = 2
        if args.dataset_name in ["reddit"]:
            args.num_neighbors = 10
        else:
            args.num_neighbors = 30
        args.dropout = 0.5
        args.sample_neighbor_strategy = "recent"
    elif args.model_name == DyGFormer.NAME:
        args.num_layers = 2
        if args.dataset_name in ["reddit"]:
            args.max_input_sequence_length = 64
            args.patch_size = 2
        else:
            args.max_input_sequence_length = 32
            args.patch_size = 1
        assert args.max_input_sequence_length % args.patch_size == 0
        if args.dataset_name in ["reddit"]:
            args.dropout = 0.2
        else:
            args.dropout = 0.1
    else:
        raise ValueError(f"Wrong value for model_name {args.model_name}!")
