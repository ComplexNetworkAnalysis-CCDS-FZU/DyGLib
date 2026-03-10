import argparse
import sys
from typing import List, Literal, Optional
import torch
from pydantic.v1 import BaseModel, Field

from models.DyGFormer import DyGFormer
from models.SignDyGFormer import SignDyGFormer

EarlyStopLiteral = Literal[
    "exist_recall", "sign_f1", "ap", "f1_macro", "f1_binary", "acc", "auc"
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
    pos_weight: float = Field(1.0, description="符号分类任务中的pos权重")

    # 模型参数
    num_heads: int = Field(2, description="注意力层中头的数量")
    num_layers: int = Field(2, description="模型层数量")
    seed: int = Field(2026)

    # 潜空间维度
    time_feat_dim: int = Field(100, description="时间编码的维度")
    position_feat_dim: int = Field(172, description="位置编码的维度")
    patch_size: int = Field(1, description="切片大小")
    channel_embedding_dim: int = Field(
        50, description="各个通道的嵌入维度（邻居共现等）"
    )
    max_input_sequence_length: int = Field(32, description="各个节点的最大输入长度")

    # 模型训练参数
    learning_rate: float = Field(0.0001)
    dropout: float = Field(0.1)
    num_epochs: int = Field(100)
    num_runs: int = Field(4)
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

    module_repeat_aware_sampler: bool = Field(False, description="启用重复感知邻居采样")

    module_repeat_aware_sign_encoder: bool = Field(
        False, description="交互对重复交互对符号影响也考虑"
    )

    module_balance_theory_encoder:bool =Field(
        True,description="平衡理论编码模块"
    )

    module_common_neighbor_aware_sampler:bool = Field(
        True,description="历史共邻居采样感知模块"
    )



    @property
    def device(self):
        return (
            f"cuda:{self.gpu}" if torch.cuda.is_available() and self.gpu >= 0 else "cpu"
        )


def get_sign_prediction_args(is_evaluation=False):
    import pydantic_argparse

    parser = pydantic_argparse.ArgumentParser(SignPredictArgs)

    try:
        args = parser.parse_typed_args()
    except:
        parser.print_help()
        sys.exit()

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
