# data_load.py
# -*- coding: utf-8 -*-
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import Dataset, DataLoader

from pars_args import args

# semantic cache: (semantic_file, num_nodes) -> (category_lookup, subclass_lookup, category_vocab, subclass_vocab)
_SEMANTIC_CACHE: Dict[Tuple[str, int], Tuple[np.ndarray, np.ndarray, int, int]] = {}


# ----------------------------
# Semantic utils
# ----------------------------
def load_semantic_info(semantic_file: str, num_nodes: int):
    key = (semantic_file, num_nodes)
    if key in _SEMANTIC_CACHE:
        return _SEMANTIC_CACHE[key]

    category_lookup = np.zeros(num_nodes, dtype=np.int64)
    subclass_lookup = np.zeros(num_nodes, dtype=np.int64)
    max_cat = 0
    max_sub = 0

    with open(semantic_file, "r") as f:
        for line in f:
            if not line.strip():
                continue
            nid, cat, sub = line.strip().split()
            nid = int(nid)
            cat = int(cat)
            sub = int(sub)

            max_cat = max(max_cat, cat)
            max_sub = max(max_sub, sub)

            # shift by 1 (0 reserved for padding)
            category_lookup[nid] = cat + 1
            subclass_lookup[nid] = sub + 1

    category_vocab = max_cat + 2  # +1 for padding, +1 for shift
    subclass_vocab = max_sub + 2

    _SEMANTIC_CACHE[key] = (category_lookup, subclass_lookup, category_vocab, subclass_vocab)
    return _SEMANTIC_CACHE[key]


def _padding_mask_from_lengths(lengths: List[int]) -> torch.Tensor:
    max_len = max(lengths) if lengths else 0
    return torch.arange(max_len).unsqueeze(0) >= torch.as_tensor(lengths).unsqueeze(1)


def _truncate_seq(seq, max_len: int, keep: str = "last"):
    if max_len is None or max_len <= 0:
        return seq
    L = len(seq)
    if L <= max_len:
        return seq
    return seq[:max_len] if keep == "first" else seq[-max_len:]


def _build_semantic_pack(
    seq_batch: List[torch.Tensor],
    category_lookup: np.ndarray,
    subclass_lookup: np.ndarray,
):
    tokens, segments, positions, lengths = [], [], [], []
    for seq in seq_batch:
        idxs = seq.tolist()
        lengths.append(len(idxs))
        tokens.append(torch.as_tensor(subclass_lookup[idxs], dtype=torch.long))
        segments.append(torch.as_tensor(category_lookup[idxs], dtype=torch.long))
        positions.append(torch.arange(len(idxs), dtype=torch.long))

    tokens = rnn_utils.pad_sequence(tokens, batch_first=True, padding_value=0)
    segments = rnn_utils.pad_sequence(segments, batch_first=True, padding_value=0)
    positions = rnn_utils.pad_sequence(positions, batch_first=True, padding_value=0)
    padding_mask = _padding_mask_from_lengths(lengths)

    return {
        "tokens": tokens,
        "segments": segments,
        "positions": positions,
        "padding_mask": padding_mask,
        "lengths": lengths,  # keep python list
    }


def _move_semantic_pack_to_device(semantic_pack, device):
    out = {}
    for k, v in semantic_pack.items():
        out[k] = v.to(device) if isinstance(v, torch.Tensor) else v
    return out


def _auto_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------
# IO
# ----------------------------
def load_poi_neighbors(poi_file: str):
    data = np.load(poi_file, allow_pickle=True)
    return data["neighbors"]


def load_traindata(train_file: str):
    """
    Required keys:
      train_list: [N] trajectories (each is a list/array of poi ids)
      train_idx:  [N] global id per trajectory
      train_pos:  [N, K] global positive candidates
      train_prob: [N, K] sampling probability per candidate (float)
    """
    data = np.load(train_file, allow_pickle=True)
    x = data["train_list"]
    train_idx = data["train_idx"]      # [N]
    train_pos = data["train_pos"]      # [N, K]
    train_prob = data["train_prob"]    # [N, K]
    return x, train_idx, train_pos, train_prob

def load_valdata(val_file: str):
    """
    支持两种验证文件：

    A) val_file 直接带轨迹: val_list / test_list / train_list
    B) val_file 是 split 文件: 只有 val_idx / test_idx 等全局索引
       此时轨迹要从 args.tra_file (bj_tra.npy) 或 args.raw_tra_file 读取。
    """
    data = np.load(val_file, allow_pickle=True)

    # ---------- Case A: val_file 自带轨迹 ----------
    if "val_list" in data:
        x = data["val_list"]
        y = data["val_idx"] if "val_idx" in data else None
        return x, y
    if "test_list" in data:
        x = data["test_list"]
        y = data["test_idx"] if "test_idx" in data else None
        return x, y
    if "train_list" in data:
        x = data["train_list"]
        y = data["train_idx"] if "train_idx" in data else None
        return x, y

    # ---------- Case B: split 文件：用全量轨迹库取 ----------
    idx_key = None
    for k in ["val_idx", "test_idx"]:
        if k in data:
            idx_key = k
            break
    if idx_key is None:
        raise KeyError(f"Cannot find trajectory list nor val_idx/test_idx in {val_file}. keys={list(data.keys())}")

    idx_global = np.asarray(data[idx_key], dtype=np.int64)

    # 你生成数据时的全量轨迹库是 TRA_FILE = bj_tra.npy
    tra_path = getattr(args, "tra_file", None)
    if tra_path is None:
        tra_path = getattr(args, "raw_tra_file", None)

    if tra_path is None:
        raise KeyError(
            f"{val_file} is a split file ({idx_key} exists), but args.tra_file/raw_tra_file is not set. "
            f"Please set it to bj_tra.npy."
        )

    tra_all = np.load(tra_path, allow_pickle=True)
    if idx_global.max(initial=-1) >= len(tra_all):
        raise IndexError(
            f"{idx_key}.max()={int(idx_global.max())} but len(tra_all)={len(tra_all)} from {tra_path}. "
            f"Trajectory base file mismatch."
        )

    x = tra_all[idx_global]
    y = idx_global
    return x, y


# def load_valdata(val_file: str):
#     """
#     Try common key names. Adjust here if你的npz字段不同。
#     Returns:
#       x: trajectories
#       y: optional labels/indices (can be None)
#     """
#     data = np.load(val_file, allow_pickle=True)

#     # trajectory
#     if "val_list" in data:
#         x = data["val_list"]
#     elif "test_list" in data:
#         x = data["test_list"]
#     elif "train_list" in data:
#         x = data["train_list"]
#     else:
#         raise KeyError(f"Cannot find trajectory list in {val_file}. keys={list(data.keys())}")

#     # label / index (optional)
#     y = None
#     for k in ["val_y", "val_idx", "test_y", "test_idx", "y_idx", "y"]:
#         if k in data:
#             y = data[k]
#             break
#     return x, y


# ----------------------------
# Dataset
# ----------------------------
class MyData(Dataset):
    """
    label 不再是一个 int，而是 (pos_global_row, prob_row, anchor_local_idx, anchor_global_idx)
    __getitem__ returns: (traj, pos_global_row, prob_row, local_idx, global_idx)
    """
    def __init__(self, data, pos_global, prob, idx_global):
        self.data = data
        self.pos_global = pos_global
        self.prob = prob
        self.idx_global = idx_global

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (self.data[idx], self.pos_global[idx], self.prob[idx], idx, self.idx_global[idx])


class EvalTrajDS(Dataset):
    """for val/test: returns (traj, y(optional), local_idx)"""
    def __init__(self, x, y=None):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        yi = None if self.y is None else self.y[i]
        return self.x[i], yi, i


# ----------------------------
# Main loaders
# ----------------------------
def TrainValueDataLoader(train_file: str, poi_file: str, batchsize: int):
    """
    Train loader with:
      - trajectory-level positive sampling (from train_pos/train_prob, global->local mapping)
      - trajectory-level negative sampling
      - poi-level positive/negative sampling (neighbors)
      - traj-internal neighbor pos/neg
      - semantic packs for anchor/pos/neg
      - optional truncation by args.max_seq_len (default 1024), keep last
    """
    semantic_lookup = load_semantic_info(args.semantic_file, args.nodes)
    args.category_vocab, args.subclass_vocab = semantic_lookup[2], semantic_lookup[3]

    train_x, train_idx_global, train_pos_global, train_prob = load_traindata(train_file)
    neighbors = load_poi_neighbors(poi_file)
    tra_len = int(train_x.shape[0])
    device = _auto_device()

    # global id -> local row
    global2local = {int(g): int(i) for i, g in enumerate(train_idx_global)}

    MAX_T = int(getattr(args, "max_seq_len", 1024))
    KEEP = "last"

    rng = np.random.default_rng(int(getattr(args, "seed", 123)))

    def _sample_pos_local(pos_global_row, prob_row, anchor_local, anchor_global):
        pos_global_row = np.asarray(pos_global_row, dtype=np.int64)
        K = int(pos_global_row.shape[0])

        prob = np.asarray(prob_row, dtype=np.float64)
        if prob.shape != (K,):
            prob = None
        else:
            prob = np.nan_to_num(prob, nan=0.0, posinf=0.0, neginf=0.0)
            prob = np.clip(prob, 0.0, None)
            s = float(prob.sum())
            prob = (prob / s) if s > 0 else None

        # try K times sampling
        for _ in range(K):
            jg = int(rng.choice(pos_global_row, p=prob)) if prob is not None else int(rng.choice(pos_global_row))
            if jg == int(anchor_global):
                continue
            jl = global2local.get(jg, None)
            if jl is not None and jl != anchor_local:
                return jl

        # fallback: sequential scan
        for jg in pos_global_row:
            jg = int(jg)
            if jg == int(anchor_global):
                continue
            jl = global2local.get(jg, None)
            if jl is not None and jl != anchor_local:
                return jl

        # final fallback: random local
        jl = int(rng.integers(0, tra_len))
        while jl == anchor_local:
            jl = int(rng.integers(0, tra_len))
        return jl

    def collate_fn_neg(data_tuple):
        # sort by traj length desc
        data_tuple.sort(key=lambda x: len(x[0]), reverse=True)

        # anchor trajs (truncated)
        data = [torch.LongTensor(_truncate_seq(sq[0], MAX_T, KEEP)) for sq in data_tuple]
        idx_list = [int(sq[3]) for sq in data_tuple]
        idx_global_list = [int(sq[4]) for sq in data_tuple]

        # sample 1 pos traj per anchor
        label_indices = []
        for sq in data_tuple:
            pos_local = _sample_pos_local(
                pos_global_row=sq[1],
                prob_row=sq[2],
                anchor_local=int(sq[3]),
                anchor_global=int(sq[4]),
            )
            label_indices.append(pos_local)

        data_label = [torch.LongTensor(_truncate_seq(train_x[d], MAX_T, KEEP)) for d in label_indices]

        # neg traj sampling
        data_neg = []
        for b in range(len(data)):
            neg = int(rng.integers(0, tra_len))
            while neg == idx_list[b] or neg == label_indices[b]:
                neg = int(rng.integers(0, tra_len))
            data_neg.append(torch.LongTensor(_truncate_seq(train_x[neg], MAX_T, KEEP)))

        data_length = [len(sq) for sq in data]
        neg_length = [len(sq) for sq in data_neg]
        label_length = [len(sq) for sq in data_label]

        # POI-level pos/neg (neighbors)
        poi_pos, poi_neg = [], []
        for traj in data:
            pos, neg = [], []
            for poi in traj:
                poi_id = int(poi)
                nb = neighbors[poi_id]
                pos_id = int(rng.integers(0, len(nb)))
                pos.append(int(nb[pos_id]))

                neg_id = int(rng.integers(0, args.nodes))
                while (neg_id in nb) or (neg_id == poi_id):
                    neg_id = int(rng.integers(0, args.nodes))
                neg.append(int(neg_id))
            poi_pos.append(torch.LongTensor(pos))
            poi_neg.append(torch.LongTensor(neg))

        # traj internal neighbor pos/neg
        traj_poi_pos, traj_poi_neg = [], []
        for traj in data:
            traj_list = traj.tolist()
            L = len(traj_list)
            pos, neg = [], []
            if L == 0:
                pos = [0]
                neg = [0]
            elif L == 1:
                pos.append(traj_list[0])
                neg_id = int(rng.integers(0, args.nodes))
                while neg_id == traj_list[0]:
                    neg_id = int(rng.integers(0, args.nodes))
                neg.append(neg_id)
            else:
                for i in range(L):
                    if i == 0:
                        pos.append(traj_list[i + 1])
                    elif i == L - 1:
                        pos.append(traj_list[i - 1])
                    else:
                        pos.append(traj_list[i - 1] if rng.random() <= 0.5 else traj_list[i + 1])

                    neg_id = int(rng.integers(0, args.nodes))
                    while neg_id in traj_list:
                        neg_id = int(rng.integers(0, args.nodes))
                    neg.append(neg_id)

            traj_poi_pos.append(torch.LongTensor(pos))
            traj_poi_neg.append(torch.LongTensor(neg))

        # semantic packs (build on CPU)
        semantic_anchor = _build_semantic_pack(data, semantic_lookup[0], semantic_lookup[1])
        semantic_pos = _build_semantic_pack(data_label, semantic_lookup[0], semantic_lookup[1])
        semantic_neg = _build_semantic_pack(data_neg, semantic_lookup[0], semantic_lookup[1])

        # pad sequences
        data = rnn_utils.pad_sequence(data, batch_first=True, padding_value=0)
        data_label = rnn_utils.pad_sequence(data_label, batch_first=True, padding_value=0)
        data_neg = rnn_utils.pad_sequence(data_neg, batch_first=True, padding_value=0)

        traj_poi_pos_tensor = rnn_utils.pad_sequence(traj_poi_pos, batch_first=True, padding_value=0)
        traj_poi_neg_tensor = rnn_utils.pad_sequence(traj_poi_neg, batch_first=True, padding_value=0)

        poi_pos_tensor = rnn_utils.pad_sequence(poi_pos, batch_first=True, padding_value=0)
        poi_neg_tensor = rnn_utils.pad_sequence(poi_neg, batch_first=True, padding_value=0)

        # move to device
        data = data.to(device)
        data_label = data_label.to(device)
        data_neg = data_neg.to(device)

        traj_poi_pos_tensor = traj_poi_pos_tensor.to(device)
        traj_poi_neg_tensor = traj_poi_neg_tensor.to(device)
        poi_pos_tensor = poi_pos_tensor.to(device)
        poi_neg_tensor = poi_neg_tensor.to(device)

        semantic_anchor_dev = _move_semantic_pack_to_device(semantic_anchor, device)
        semantic_pos_dev = _move_semantic_pack_to_device(semantic_pos, device)
        semantic_neg_dev = _move_semantic_pack_to_device(semantic_neg, device)

        return (
            data, data_neg, data_label,
            data_length, neg_length, label_length,
            traj_poi_pos_tensor, traj_poi_neg_tensor,
            poi_pos_tensor, poi_neg_tensor,
            semantic_anchor_dev, semantic_pos_dev, semantic_neg_dev,
            # idx_global_list,  # optional: keep global ids if you need
        )

    dataset = DataLoader(
        MyData(train_x, train_pos_global, train_prob, train_idx_global),
        batch_size=batchsize,
        shuffle=True,
        collate_fn=collate_fn_neg,
    )
    return dataset


def TrainDataValLoader(train_file: str, batchsize: int):
    """
    Eval loader on train set (no pos/neg sampling).
    Returns:
      data, idx_global_list, data_length, idx_list, semantic_pack
    """
    semantic_lookup = load_semantic_info(args.semantic_file, args.nodes)
    args.category_vocab, args.subclass_vocab = semantic_lookup[2], semantic_lookup[3]
    device = _auto_device()

    train_x, train_idx_global, _, _ = load_traindata(train_file)

    MAX_T = int(getattr(args, "max_seq_len", 1024))
    KEEP = "last"

    # dummy pos/prob just to reuse MyData signature
    dummy_pos = np.zeros((len(train_x), 1), dtype=np.int64)
    dummy_prob = np.zeros((len(train_x), 1), dtype=np.float32)

    def collate_fn_eval(data_tuple):
        # data_tuple: (traj, pos_row, prob_row, local_idx, global_idx)
        data_tuple.sort(key=lambda x: len(x[0]), reverse=True)

        data = [torch.LongTensor(_truncate_seq(sq[0], MAX_T, KEEP)) for sq in data_tuple]
        idx_list = [int(sq[3]) for sq in data_tuple]
        idx_global_list = [int(sq[4]) for sq in data_tuple]
        data_length = [len(sq) for sq in data]

        semantic_pack = _build_semantic_pack(data, semantic_lookup[0], semantic_lookup[1])

        data = rnn_utils.pad_sequence(data, batch_first=True, padding_value=0).to(device)
        semantic_pack_dev = _move_semantic_pack_to_device(semantic_pack, device)

        return data, idx_global_list, data_length, idx_list, semantic_pack_dev

    dataset = DataLoader(
        MyData(train_x, dummy_pos, dummy_prob, train_idx_global),
        batch_size=batchsize,
        shuffle=False,
        collate_fn=collate_fn_eval,
    )
    return dataset


def ValValueDataLoader(val_file: str, batchsize: int):
    """
    Val loader (no pos/neg sampling).
    Returns:
      data, labels(optional), data_length, idx_list, semantic_pack
    """
    semantic_lookup = load_semantic_info(args.semantic_file, args.nodes)
    args.category_vocab, args.subclass_vocab = semantic_lookup[2], semantic_lookup[3]
    device = _auto_device()

    val_x, val_y = load_valdata(val_file)

    MAX_T = int(getattr(args, "max_seq_len", 1024))
    KEEP = "last"

    def collate_fn_eval(batch):
        # batch: (traj, y, local_idx)
        batch.sort(key=lambda x: len(x[0]), reverse=True)

        data = [torch.LongTensor(_truncate_seq(b[0], MAX_T, KEEP)) for b in batch]
        labels = [b[1] for b in batch]
        idx_list = [int(b[2]) for b in batch]
        data_length = [len(sq) for sq in data]

        semantic_pack = _build_semantic_pack(data, semantic_lookup[0], semantic_lookup[1])

        data = rnn_utils.pad_sequence(data, batch_first=True, padding_value=0).to(device)
        semantic_pack_dev = _move_semantic_pack_to_device(semantic_pack, device)

        return data, labels, data_length, idx_list, semantic_pack_dev

    dataset = DataLoader(
        EvalTrajDS(val_x, val_y),
        batch_size=batchsize,
        shuffle=False,
        collate_fn=collate_fn_eval,
    )
    return dataset


def TestValueDataLoader(test_file: str, batchsize: int):
    """
    Test loader (same as ValValueDataLoader).
    """
    return ValValueDataLoader(test_file, batchsize)
