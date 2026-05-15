import itertools
from typing import Literal

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def print_heatmap(
    data: pd.DataFrame,
    mask,
    xlabel: list,
    ylabel: list,
    x_name: str,
    y_name: str,
    dataset_name: str,
    metric,
    task: str,
):
    plt.figure()
    vmin=data[data!=0].min()
    print(f"min non-zero value is : {vmin}")
    sns.heatmap(data,annot=True,cmap='crest',xticklabels=xlabel,yticklabels=ylabel,fmt=".4f",vmin=vmin,mask=mask)
    # plt.plot(xlabel, data[:, 3], marker="o")
    plt.title("heatmap")
    plt.xlabel(x_name, fontsize=14)
    # plt.xticks(xaxis)
    # plt.yticks(yaxis)
    plt.ylabel(y_name, fontsize=14)
    plt.title(f"{dataset_name}({metric})")
    plt.savefig(f"figs/{task}-{dataset_name}-img.pdf", dpi=300, format="pdf")


def dataset_collect(data: pd.Series,org_df, metric):
    # LF 0~20 步长5
    # NN 0~100 步长20
     # 从实际数据中获取唯一的参数值，并排序
    lf_values = sorted(data['LF'].unique())
    nn_values = sorted(data['NN'].unique())
    
    print(f"实际 LF 值: {lf_values}")
    print(f"实际 NN 值: {nn_values}")
    
    # 创建结果数组
    array = np.zeros((len(lf_values), len(nn_values)), np.float64)
    mask = np.zeros((len(lf_values), len(nn_values)), np.bool)
    gs = data.groupby(["NN", "LF"])

    # array = np.zeros((5, 6), np.float64)
    gs = data.groupby(["NN", "LF"])

    for lf, nn in itertools.product(
        enumerate(lf_values), enumerate(nn_values)
    ):
        lf_idx, lf = lf
        nn_idx, nn = nn
        value = 0
        need_mask = True  
      

        try:
            key = gs.groups[(nn, lf)]

            lines = org_df.iloc[key]
        except KeyError as _:
            print(f"Data at NN: {nn}, LF: {lf} missing")

        else:
            if len(lines) == 1:
                need_mask = False
                value = lines.iloc[0][metric]
        array[lf_idx, nn_idx] = value
        mask[lf_idx,nn_idx] = need_mask

    return array,mask, lf_values, nn_values


def load_collect_result(file_name, task):
    df = pd.read_csv(file_name)
    grouped = df.groupby("Dataset")

    for k, idxes in grouped.groups.items():
        print(f"now collect dataset {k}")
        rr = df.iloc[idxes]

        metrix = "AUC" if task =="sign" else "F1_W"
        data,mask, ylabel, xlabel = dataset_collect(rr,df,metrix)

        print_heatmap(data,mask, xlabel, ylabel, "Number Neighbor", "Sampling-aware Node Looking Forward", dataset_name=k, task=task,metric=metrix)

from pydantic.v1 import BaseModel,Field
from pydantic_argparse import ArgumentParser,argparse
class Params(BaseModel):
    task:Literal["linksign","sign"] = Field(description="收集任务类型")

    @property
    def filename(self):
        if self.task =="linksign":
            return "./param-linksign.csv"
        else:
            return "./param-sign.csv"
if __name__ == "__main__":
    args = ArgumentParser(Params)
    args = args.parse_typed_args()

    # # 生成 10x10 的随机数据
    # data = np.random.rand(10, 10)

    # # 将 NumPy 数据转换为 Pandas DataFrame，并给行和列指定标签
    # data_frame = pd.DataFrame(data,
    #                       index=[f'Row{i}' for i in range(1, 11)],
    #                       columns=[f'Col{i}' for i in range(1, 11)])

    # print_heatmap(data_frame,[0,5,10,15,20],[0,20,40,60,80,100])

    file_name = args.filename
    load_collect_result(file_name, args.task)
    # df = pd.read_csv(file_name)
    # grouped = df.groupby("Dataset")
    # rr = df.iloc[grouped.groups["RedditHyperlinkBody"]]

    # data,xlabel,ylabel = dataset_collect(rr,"F1_B")

    # print_heatmap(data,xlabel,ylabel,"LR","NN","RedditHyperlinkBody")

    # print(rr,rr["NN"].max(),rr["LF"].max())

    # gr = rr.groupby(["NN","LF"])

    # rc = df.iloc[gr.groups[(100,5)]]

    # print(rc,len(rc))
    # print(list(range(0,21,5)))


# 不同数据集的不同参数实