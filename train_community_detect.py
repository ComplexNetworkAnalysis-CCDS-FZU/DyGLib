import logging
import os
import time

import torch
from tqdm import tqdm
from models.DyGFormer import DyGFormer
from utils.utils import get_neighbor_sampler, set_random_seed


DATASET = "<DataSET>"
EPOCHS = 100


def load_dataset(datasetName: str, valRatio: float, testRatio: float):
    """加载数据集

    Args:
        datasetName (str): 数据集名称
        valRatio (float): 划分验证集的比例
        testRatio (float): 划分测试集部分
    """
    pass


if __name__ == "__main__":

    # 加载数据集
    node_features, edge_features, full_data, train_data, val_data, test_data = (
        load_dataset(DATASET, 0.1, 0.2)
    )

    # 对历史邻居节点进行编码采样
    # TODO: 接入实现
    full_neighbor_sampler = get_neighbor_sampler(data=full_data)

    # 将数据集转换为torch的DataLoader
    train_idx_dataloader = None
    val_idx_dataloader = None
    test_idx_dataloader = None

    set_random_seed(114514)

    # 训练开始前准备

    # 日志部分初始化
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    os.makedirs(f"./logs/DyGFormer/{DATASET}/communityDetect", exist_ok=True)

    # 导出到文件的日志
    fh = logging.FileHandler(
        f"./logs/DyGFormer/{DATASET}/communityDetect/{str(time.time())}.log"
    )
    fh.setLevel(logging.DEBUG)

    # 在命令行显示的日志
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)

    # 日志格式化
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # 注册日志Handler
    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info("Start Training")

    # 模型准备

    dynamic_backbone = DyGFormer(
        node_raw_features=node_features,
        edge_raw_features=edge_features,
        neighbor_sampler=full_neighbor_sampler,
        time_feat_dim=172,
        channel_embedding_dim=200,
        patch_size=20,
        num_layers=8,
        num_heads=8,
        dropout=0.1,
        max_input_sequence_length=512,
        device="cuda",
    )
    # TODO: 下游的社区发现任务模型
    community_detect = None

    model = torch.nn.Sequential(dynamic_backbone, community_detect)

    # 早停策略 TODO

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 损失函数 TODO
    loss_function = None

    # set the dynamic_backbone in evaluation mode
    # 使用预训练模型
    model[0].eval()

    for epoch in range(EPOCHS):
        model[1].train()
        # 训练时，提供邻居采样
        model[0].set_neighbor_sampler(full_neighbor_sampler)

        # store train losses, trues and predicts
        train_total_loss, train_y_trues, train_y_predicts = 0.0, [], []
        train_idx_data_loader_tqdm = tqdm(train_idx_dataloader, ncols=120)
        for batch_idx, train_data_indices in enumerate(train_idx_data_loader_tqdm):
            train_data_indices = train_data_indices.numpy()
            (
                batch_src_node_ids,
                batch_dst_node_ids,
                batch_node_interact_times,
                batch_edge_ids,
                batch_labels,
            ) = (
                train_data.src_node_ids[train_data_indices],
                train_data.dst_node_ids[train_data_indices],
                train_data.node_interact_times[train_data_indices],
                train_data.edge_ids[train_data_indices],
                train_data.labels[train_data_indices],
            )

            with torch.no_grad():
                # get temporal embedding of source and destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_src_node_embeddings, batch_dst_node_embeddings = model[
                    0
                ].compute_src_dst_node_temporal_embeddings(
                    src_node_ids=batch_src_node_ids,
                    dst_node_ids=batch_dst_node_ids,
                    node_interact_times=batch_node_interact_times,
                    # TODO: 采样邻居数量与时间间隔为超参数，需要实验考虑
                    num_neighbors=30,
                    time_gap=1000,
                )

            predict = model[1](batch_src_node_embeddings)
            # TODO: 社区发现的标签
            label = None

            loss = loss_function(predict, label)

            train_total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_idx_data_loader_tqdm.set_description(
                f"Epoch: {epoch + 1}, train for the {batch_idx + 1}-th batch, train loss: {loss.item()}"
            )
