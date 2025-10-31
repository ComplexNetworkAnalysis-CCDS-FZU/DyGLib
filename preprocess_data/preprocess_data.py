import numpy as np
import pandas as pd
from pathlib import Path
from pandas.testing import assert_frame_equal
from distutils.dir_util import copy_tree
from pydantic.v1 import BaseModel, Field
import pydantic_argparse
from typing import Literal


def preprocess(
    dataset_name: str,
    unsigned: bool = False,
    skip_first_line: bool = True,
    sep: str = ",",
    swap_time_sign: bool = False,
):
    """
    read the original data file and return the DataFrame that has columns ['u', 'i', 'ts','sign', 'label', 'idx']
    :param dataset_name: str, dataset name
    :return:
    """
    u_list, i_list, ts_list, sign_list, label_list = [], [], [], [], []
    feat_l = []
    idx_list = []

    ts_idx, sign_idx = (3, 2) if swap_time_sign else (2, 3)

    with open(dataset_name) as f:
        # skip the first line
        if skip_first_line:
            s = next(f)
        lines = f.readlines()
        lines.sort(key=lambda x: float(x.strip().split(sep)[-1]))
        previous_time = -1
        import tqdm

        for idx, line in tqdm.tqdm(enumerate(lines)):
            e = line.strip().split(sep)
            # user_id
            u = int(e[0])
            # item_id
            i = int(e[1])

            # timestamp
            ts = float(e[ts_idx])
            # check whether time in ascending order
            assert ts >= previous_time
            previous_time = ts
            # interactive sign
            sign = (
                1
                if unsigned
                else (
                    0
                    if int(e[sign_idx]) == 0
                    else int(e[sign_idx]) // abs(int(e[sign_idx]))
                )
            )
            # state_label
            label = (
                0.0
                if len(e) <= (3 if unsigned else 4)
                else float(e[3 if unsigned else 4])
            )

            # edge features
            sign_feat = [] if unsigned else [float(e[sign_idx])]
            sign_feat.extend([float(x) for x in e[4 if unsigned else 5 :]])
            feat = np.array(sign_feat)

            u_list.append(u)
            i_list.append(i)
            ts_list.append(ts)
            sign_list.append(sign)
            label_list.append(label)
            # edge index
            idx_list.append(idx)

            feat_l.append(feat)
    return pd.DataFrame(
        {
            "u": u_list,
            "i": i_list,
            "ts": ts_list,
            "sign": sign_list,
            "label": label_list,
            "idx": idx_list,
        }
    ), np.array(feat_l)


def reindex(df: pd.DataFrame, bipartite: bool = True):
    """
    reindex the ids of nodes and edges
    :param df: DataFrame
    :param bipartite: boolean, whether the graph is bipartite or not
    :return:
    """
    new_df = df.copy()
    if bipartite:
        # check the ids of users and items
        assert df.u.max() - df.u.min() + 1 == len(df.u.unique())
        assert df.i.max() - df.i.min() + 1 == len(df.i.unique())
        assert df.u.min() == df.i.min() == 0

        # if bipartite, discriminate the source and target node by unique ids (target node id is counted based on source node id)
        upper_u = df.u.max() + 1
        new_i = df.i + upper_u

        new_df.i = new_i

    # make the id start from 1
    new_df.u += 1
    new_df.i += 1
    new_df.idx += 1

    return new_df


def preprocess_data(
    dataset_name: str,
    bipartite: bool = True,
    node_feat_dim: int = 172,
    unsigned: bool = False,
    skip_first_line: bool = True,
    sep: str = ",",
    swap_time_sign: bool = False,
):
    """
    preprocess the data
    :param dataset_name: str, dataset name
    :param bipartite: boolean, whether the graph is bipartite or not
    :param node_feat_dim: int, dimension of node features
    :return:
    """
    Path("../processed_data/{}/".format(dataset_name)).mkdir(
        parents=True, exist_ok=True
    )
    PATH = "../DG_data/{}/{}.csv".format(dataset_name, dataset_name)
    OUT_DF = "../processed_data/{}/ml_{}.csv".format(dataset_name, dataset_name)
    OUT_FEAT = "../processed_data/{}/ml_{}.npy".format(dataset_name, dataset_name)
    OUT_NODE_FEAT = "../processed_data/{}/ml_{}_node.npy".format(
        dataset_name, dataset_name
    )

    df, edge_feats = preprocess(
        PATH, unsigned, skip_first_line, sep, swap_time_sign=swap_time_sign
    )
    new_df = reindex(df, bipartite)

    # edge feature for zero index, which is not used (since edge id starts from 1)
    empty = np.zeros(edge_feats.shape[1])[np.newaxis, :]
    # Stack arrays in sequence vertically(row wise),
    edge_feats = np.vstack([empty, edge_feats])

    # node features with one additional feature for zero index (since node id starts from 1)
    max_idx = max(new_df.u.max(), new_df.i.max())
    node_feats = np.zeros((max_idx + 1, node_feat_dim))

    print("number of nodes ", node_feats.shape[0] - 1)
    print("number of node features ", node_feats.shape[1])
    print("number of edges ", edge_feats.shape[0] - 1)
    print("number of edge features ", edge_feats.shape[1])

    new_df.to_csv(OUT_DF)  # edge-list
    np.save(OUT_FEAT, edge_feats)  # edge features
    np.save(OUT_NODE_FEAT, node_feats)  # node features


def check_data(dataset_name: str):
    """
    check whether the processed datasets are identical to the given processed datasets
    :param dataset_name: str, dataset name
    :return:
    """
    # original data paths
    origin_OUT_DF = "../DG_data/{}/ml_{}.csv".format(dataset_name, dataset_name)
    origin_OUT_FEAT = "../DG_data/{}/ml_{}.npy".format(dataset_name, dataset_name)
    origin_OUT_NODE_FEAT = "../DG_data/{}/ml_{}_node.npy".format(
        dataset_name, dataset_name
    )

    # processed data paths
    OUT_DF = "../processed_data/{}/ml_{}.csv".format(dataset_name, dataset_name)
    OUT_FEAT = "../processed_data/{}/ml_{}.npy".format(dataset_name, dataset_name)
    OUT_NODE_FEAT = "../processed_data/{}/ml_{}_node.npy".format(
        dataset_name, dataset_name
    )

    # Load original data
    origin_g_df = pd.read_csv(origin_OUT_DF)
    origin_e_feat = np.load(origin_OUT_FEAT)
    origin_n_feat = np.load(origin_OUT_NODE_FEAT)

    # Load processed data
    g_df = pd.read_csv(OUT_DF)
    e_feat = np.load(OUT_FEAT)
    n_feat = np.load(OUT_NODE_FEAT)

    assert_frame_equal(origin_g_df, g_df)
    # check numbers of edges and edge features
    assert (
        origin_e_feat.shape == e_feat.shape
        and origin_e_feat.max() == e_feat.max()
        and origin_e_feat.min() == e_feat.min()
    )
    # check numbers of nodes and node features
    assert (
        origin_n_feat.shape == n_feat.shape
        and origin_n_feat.max() == n_feat.max()
        and origin_n_feat.min() == n_feat.min()
    )


class PreprocessArgs(BaseModel):
    dataset_name: Literal[
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
    ] = Field(default="wikipedia", description="Dataset name")
    node_feat_dim: int = Field(172, description="Number of node raw features")
    unsigned: bool = Field(False, description="Unsigned Graph")


parser = pydantic_argparse.ArgumentParser(
    model=PreprocessArgs, description="Interface for preprocessing datasets"
)
args = parser.parse_typed_args()
print(type(args))


print(f"preprocess dataset {args.dataset_name}...")
if args.dataset_name in ["enron", "SocialEvo", "uci"]:
    Path("../processed_data/{}/".format(args.dataset_name)).mkdir(
        parents=True, exist_ok=True
    )
    copy_tree(
        "../DG_data/{}/".format(args.dataset_name),
        "../processed_data/{}/".format(args.dataset_name),
    )
    print(
        f"the original dataset of {args.dataset_name} is unavailable, directly use the processed dataset by previous works."
    )
else:
    # bipartite dataset
    if args.dataset_name in ["wikipedia", "reddit", "mooc", "lastfm", "myket"]:
        preprocess_data(
            dataset_name=args.dataset_name,
            bipartite=True,
            node_feat_dim=args.node_feat_dim,
            unsigned=True,
        )
    elif args.dataset_name in ["WikiVote"]:
        preprocess_data(
            dataset_name=args.dataset_name,
            bipartite=False,
            node_feat_dim=args.node_feat_dim,
            unsigned=False,
        )
    elif args.dataset_name in ["BitcoinAlpha", "BitcoinOTC"]:
        preprocess_data(
            dataset_name=args.dataset_name,
            bipartite=False,
            node_feat_dim=args.node_feat_dim,
            skip_first_line=False,
            unsigned=False,
            swap_time_sign=True,
        )
    else:
        preprocess_data(
            dataset_name=args.dataset_name,
            bipartite=False,
            node_feat_dim=args.node_feat_dim,
        )
    print(f"{args.dataset_name} is processed successfully.")

    if args.dataset_name not in ["myket"]:
        check_data(args.dataset_name)
    print(f"{args.dataset_name} passes the checks successfully.")
