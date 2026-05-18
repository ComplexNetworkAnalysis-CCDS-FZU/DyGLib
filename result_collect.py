"""
参数实验结果整理脚本
支持自定义指标提取
"""

from enum import Enum
import json
import os
from typing import Dict, List, Literal, Optional
import pandas as pd
from pathlib import Path
import re
from pydantic.v1 import Field, BaseModel
import pydantic_argparse

class CollectTy(Enum):
    PARAM = "param"
    ABLATION = "ablation"
    MULTI_SEED = "multiseed"
    PROFILER = "profiler"

class Configure(BaseModel):
    root: str = Field(default=os.getcwd(), description="实验结果记录的目录")
    output: str = Field(default="param_search", description="输出文件名称")
    metrics: Optional[List[str]] = Field(default=None, description="需求的评价指标")
    task: Literal["sign", "linksign"] = Field(description="要关注的任务")
    collect_type: CollectTy = Field(CollectTy.PARAM, description="收集类型")
    profiler_modules: Optional[List[str]] = Field(
        default=None,
        description="筛选要收集的 profiler 模块，None 表示全部 9 个模块",
    )
    # ablation: bool = Field(False, description="收集消融实验结果")
    # multiseed: bool = Field(False, description="是否包含多随机种子结果")

    @property
    def task_metrics(self):
        task_default_metric = {
            "sign": ["ap", "f1_weighted", "f1_binary", "auc", "acc"],
            "linksign": ["ap", "f1_wt", "f1_mac", "f1_mic", "auc", "acc"],
        }

        met = task_default_metric[self.task]
        if self.metrics is not None:
            met.extend(self.metrics)
        return met

    @property
    def task_dir(self):
        task_dirs = {"sign": "LinkSign", "linksign": "SignLinkPrediction"}

        return task_dirs[self.task]

    @property
    def model(self):
        return "SignDyGFormer"

    @property
    def primary_metric(self):
        task_metric = {"sign": "f1_binary", "linksign": "f1_mac"}

        return task_metric[self.task]


METRIC_ALIAS = {
    # Sign Prediction 任务名称 -> 标准名称
    "ap": "AP",
    "f1_weighted": "F1_W",
    "f1_macro": "F1_M",
    "f1_binary": "F1_B",
    "auc": "AUC",
    "acc": "ACC",
    # Link & Sign Prediction 任务名称 -> 标准名称
    "f1_mac": "F1_M",  # 统一宏平均F1
    "f1_wt": "F1_W",  # 统一加权F1
    "f1_mic": "F1_MIC",  # 微平均F1（可选）
    "exist_f1": "Exist_F1",
    "exist_recall": "Exist_Rec",
    "exist_precision": "Exist_Prec",
    "sign_f1": "Sign_F1",
}


def parse_params(filename: str):
    """解析 NN 和 LF 参数，检查所有模块启用"""
    name = filename.replace(".json", "")

    # 检查所有模块是否启用
    if not all(tag in name for tag in ["RAS-E", "RASE-E", "BTE-E", "CNAS-E"]):
        return None
    # 提取 NN 和 LF

    nn_match = re.search(r"NN-(\d+)", name)
    lf_match = re.search(r"LF-(\d+)", name)
    seed_match = re.search(r"seed(\d+)", name)

    if not nn_match or not lf_match:
        return None
    result= {"NN": int(nn_match.group(1)), "LF": int(lf_match.group(1))}
    if seed_match:
        result["seed"] = int(seed_match.group(1))
    return result

def parse_ablation(filename: str):

    name = filename.replace(".json", "")

    if not all(tag in name for tag in ["NN-Best", "LF-Best"]):
        return None

    model_status = re.search(r".RAS-([DE]).RASE-([DE]).BTE-([DE]).CNAS-([DE])", name)
    seed_match = re.search(r"seed(\d+)", name)
    if not model_status:
        return None
    reuslt =  {
        "RAS": model_status.group(1),
        "RASE": model_status.group(2),
        "BTE": model_status.group(3),
        "CNAS": model_status.group(4),
    }

    if seed_match:
        reuslt["seed"] = int(seed_match.group(1))

    return reuslt

def parse_multiseed(filename: str):
    name = filename.replace(".json", "")
    seed_match = re.search(r"seed(\d+)", name)
    if not seed_match or seed_match.group(1) == "0":
        return None
    return int(seed_match.group(1))


def parse_profiler_seed(filename: str):
    """从 profiler 文件名中提取 seed 编号，seed0 直接过滤"""
    name = filename.replace("-profiler.json", "").replace(".json", "")
    seed_match = re.search(r"seed(\d+)", name)
    if not seed_match or seed_match.group(1) == "0":
        return None
    return int(seed_match.group(1))


def extract_profiler_metrics(json_path: Path, modules: Optional[List[str]] = None):
    """提取 profiler JSON 中各模块耗时统计，ns→ms，计算占比"""
    try:
        with open(json_path, "r") as f:
            data = json.load(f)

        # 筛选模块
        all_modules = list(data.keys())
        target_modules = modules if modules is not None else all_modules

        # 收集 mean_ns 用于计算占比
        module_means = {}
        for mod in target_modules:
            if mod in data:
                module_means[mod] = data[mod]["mean_ns"]

        total_mean_ns = sum(module_means.values())

        result = {}
        for mod in target_modules:
            if mod not in data:
                continue
            stats = data[mod]
            mean_ms = stats["mean_ns"] / 1e6
            std_ms = stats["std_ns"] / 1e6
            pct = (module_means[mod] / total_mean_ns * 100) if total_mean_ns > 0 else 0.0

            # 用下划线替换模块名中的空格和逗号，作为合法的列名
            safe_name = mod.replace(" ", "_").replace(",", "")

            result[f"{safe_name}_mean_ms"] = round(mean_ms, 6)
            result[f"{safe_name}_std_ms"] = round(std_ms, 6)
            result[f"{safe_name}_pct"] = round(pct, 2)

        return result
    except Exception as e:
        print(f"错误: {json_path}: {e}")
        return None


def profiler_collect(json_file):
    """收集单个 profiler JSON 文件的数据"""
    seed = parse_profiler_seed(json_file.name)
    if seed is None:
        return {}
    metrics = extract_profiler_metrics(json_file, args.profiler_modules)
    if metrics is None:
        return {}

    return {
        "Dataset": dataset_name,
        "seed": seed,
        **metrics,
    }


def aggregate_profiler(records: List[Dict]):
    """按 Dataset 聚合 profiler 数据，计算跨 seed 均值、标准差和占比"""
    df = pd.DataFrame(records)

    if df.empty:
        print("警告: 无数据可聚合")
        return pd.DataFrame()

    # 只对 _mean_ms 列做跨 seed 聚合
    mean_ms_cols = [c for c in df.columns if c.endswith("_mean_ms")]
    agg_spec = {c: ["mean", "std"] for c in mean_ms_cols}

    grouped = df.groupby("Dataset")
    aggregated = grouped.agg(agg_spec)
    aggregated.columns = [
        f"{col[0]}_{col[1]}" for col in aggregated.columns
    ]
    aggregated = aggregated.reset_index()

    # 重命名：{module}_mean_ms_mean → {module}_mean_ms, {module}_mean_ms_std → {module}_std_ms
    rename_map = {}
    for c in aggregated.columns:
        if c.endswith("_mean_ms_mean"):
            rename_map[c] = c.replace("_mean_ms_mean", "_mean_ms")
        elif c.endswith("_mean_ms_std"):
            rename_map[c] = c.replace("_mean_ms_std", "_std_ms")
    aggregated.rename(columns=rename_map, inplace=True)

    # 重新计算 _pct（基于跨 seed 均值）
    final_mean_cols = [c for c in aggregated.columns if c.endswith("_mean_ms")]
    for row_idx in range(len(aggregated)):
        total_mean = sum(
            aggregated.loc[row_idx, col] if pd.notna(aggregated.loc[row_idx, col]) else 0
            for col in final_mean_cols
        )
        for mean_col in final_mean_cols:
            pct_col = mean_col.replace("_mean_ms", "_pct")
            mean_val = aggregated.loc[row_idx, mean_col]
            aggregated.loc[row_idx, pct_col] = (
                round(mean_val / total_mean * 100, 2) if total_mean > 0 and pd.notna(mean_val) else 0.0
            )

    # 列排序：Dataset, 然后每模块三列 (mean_ms, std_ms, pct)
    module_prefixes = list(dict.fromkeys(
        c.replace("_mean_ms", "").replace("_std_ms", "").replace("_pct", "")
        for c in aggregated.columns if c != "Dataset"
    ))
    ordered_cols = ["Dataset"]
    for prefix in sorted(module_prefixes):
        ordered_cols.extend([f"{prefix}_mean_ms", f"{prefix}_std_ms", f"{prefix}_pct"])
    aggregated = aggregated[[c for c in ordered_cols if c in aggregated.columns]]

    return aggregated


def extract_metrics(json_path: Path, metrics_list: list[str]):
    """提取指定的 test metrics"""
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        tm = data.get("test metrics", {})

        # 提取指定指标，缺失则设为 None
        result = {}
        for m in metrics_list:
            val = tm.get(m)
            std_name = METRIC_ALIAS[m]
            result[std_name] = float(val) if val is not None else None
        return result
    except Exception as e:
        print(f"错误: {json_path}: {e}")
        return None


def param_collect(json_file):
    params = parse_params(json_file.name)
    if params is None:
        return {}
    metrics = extract_metrics(json_file, args.task_metrics)

    if metrics is None:
        return {}

    return {
        "Dataset": dataset_name,
        "NN": params["NN"],
        "LF": params["LF"],
        **metrics,
    }


def ablation_collect(json_file):
    params = parse_ablation(json_file.name)
    if params is None:
        return {}
    metrics = extract_metrics(json_file, args.task_metrics)
    if metrics is None:
        return {}

    return {
        "Dataset": dataset_name,
        **params,
        **metrics,
    }

def multiseed_collect(json_file):
    seed = parse_multiseed(json_file.name)
    if seed is None:
        return {}
    metrics = extract_metrics(json_file, args.task_metrics)
    if metrics is None:
        return {}

    return {
        "Dataset": dataset_name,
        "seed": seed,
        **metrics,
    }

def aggregate_multiseed(records: List[Dict], group_keys: List[str]):
    """对多随机种子结果进行聚合，计算平均值和标准差"""

    df = pd.DataFrame(records)
    metric_cols = [c for c in df.columns 
               if c not in group_keys and c != "seed"]
    metric_cols = [c for c in metric_cols 
               if df[c].dtype in ['float64', 'float32', 'int64', 'int32']]
    
    grouped = df.groupby(group_keys)
    agg_keys = {m: ['mean', 'std','count'] for m in metric_cols}

    aggregated = grouped.agg(agg_keys)

    aggregated.columns = [f"{col[0]}_{col[1]}" for col in aggregated.columns]
    aggregated = aggregated.reset_index()

    for col in metric_cols:
        mean_col = f"{col}_mean"
        std_col = f"{col}_std"

        if mean_col in aggregated.columns and std_col in aggregated.columns:
            aggregated[f"{col}_summary"] = aggregated.apply(
            lambda row: f"{row[mean_col]:.4f} ± {row[std_col]:.4f}"if pd.notna(row[mean_col]) and pd.notna(row[std_col]) else str(row[mean_col]), axis=1
        )
            
    return aggregated

if __name__ == "__main__":
    parser = pydantic_argparse.ArgumentParser(Configure)
    args = parser.parse_typed_args()

    print(f"task: {args.task},type: {args.collect_type}, 提取参数: 【{','.join(args.task_metrics)}】")

    root = Path(args.root)
    records = []

    result_dir = root / args.task_dir / args.model

    for dataset_dir in result_dir.iterdir():
        print(f"now at dir {dataset_dir}")
        if not dataset_dir.is_dir():
            continue
        dataset_name = dataset_dir.name

        if args.collect_type == CollectTy.PROFILER:
            # Profiler JSON 在 {dataset_name}/{save_model_name}/ 子目录下
            for sub_dir in dataset_dir.iterdir():
                if not sub_dir.is_dir():
                    continue
                for json_file in sub_dir.glob("*profiler*.json"):
                    print(f"now at profiler file: {json_file}")
                    record = profiler_collect(json_file)
                    if record:
                        records.append(record)
        else:
            for json_file in dataset_dir.glob("*.json"):
                print(f"now at file : {json_file}")
                if args.collect_type == CollectTy.ABLATION:
                    record = ablation_collect(json_file)
                elif args.collect_type == CollectTy.MULTI_SEED:
                    record = multiseed_collect(json_file)
                else:
                    record = param_collect(json_file)

                if record:
                    records.append(record)

    if not records:
        print("未找到符合条件的实验结果")
        os._exit(1)
    df = pd.DataFrame(records)
    df = df.sort_values(["Dataset"])

    # 保存原始数据
    df.to_csv(f"{args.output}-{args.task}-{args.collect_type.value}.csv", index=False, float_format="%.4f")
    print(f"\n保存: {args.output}-{args.task}.csv ({len(df)} 条记录)")

    if args.collect_type == CollectTy.PROFILER:
        agg = aggregate_profiler(records)
        if agg.empty:
            print("聚合后无数据")
            os._exit(1)

        agg_file = f"{args.output}-{args.task}-{args.collect_type.value}-agg.csv"
        agg.to_csv(agg_file, index=False, float_format="%.4f")
        print(f"保存聚合结果: {agg_file} ({len(agg)} 条记录)")

    elif args.collect_type == CollectTy.MULTI_SEED:
        group_keys = ["Dataset"]

        agg = aggregate_multiseed(records, group_keys)

        agg.to_csv(f"{args.output}-{args.task}-{args.collect_type.value}-agg.csv", index=False, float_format="%.4f")
        print(f"\n保存聚合结果: {args.output}-{args.task}-{args.collect_type.value}-agg ({len(agg)} 条记录)")

    else:
        # 打印最优参数（按第一个指标）
        primary_metric = METRIC_ALIAS[args.primary_metric]
        print(f"\n=== 各数据集最优参数 ({primary_metric}) ===")
        for dataset in df["Dataset"].unique():
            sub = df[df["Dataset"] == dataset]
            best = sub.loc[sub[primary_metric].idxmax()]
            if args.collect_type == CollectTy.ABLATION:
                print(
                    f"\n{dataset}: RAS={best['RAS']}, RASE={best['RASE']}, BTE={best['BTE']}, CNAS={best['CNAS']}"
                )
            else:
                print(f"\n{dataset}: NN={best['NN']}, LF={best['LF']}")
            for m in args.task_metrics:
                m = METRIC_ALIAS[m]
                print(f"  {m}={best[m]:.4f}")
