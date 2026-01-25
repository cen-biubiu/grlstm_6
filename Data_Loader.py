from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import Dataset, DataLoader

from pars_args import args

import sys
import os

sys.path.append(os.path.dirname(__file__))

# semantic cache
_SEMANTIC_CACHE: Dict[Tuple[str, int], Tuple[np.ndarray, np.ndarray, int, int]] = {}


# ==================== 增强器定义 ====================

class TrajectoryAugmentor:
    """
    四种增强策略:
    (1) 关系边mask - 仅mask类别2
    (2) 空间抖动 - 替换POI为邻近POI (只改节点特征,不改边)
    (3) 时间重采样 - 固定间隔降采样
    (4) 子序列提取 - 提取连续子序列
    """

    def __init__(
            self,
            topk_graph: Dict[int, List[Tuple[int, int]]],
            neighbors_spatial: np.ndarray,
            num_nodes: int,
            seed: int = 42
    ):
        self.topk_graph = topk_graph
        self.neighbors_spatial = neighbors_spatial
        self.num_nodes = num_nodes
        self.rng = np.random.default_rng(seed)

    def augment_strategy_1_edge_mask(
            self,
            traj: np.ndarray,
            # 注意: 边mask在图层面操作,这里返回原始轨迹
            # 实际mask在模型的图构建阶段进行
    ) -> np.ndarray:
        """
        策略1: 关系边mask
        注意: 这个增强需要在图构建时mask边,这里只返回原始轨迹
        实际实现在模型forward中
        """
        return traj.copy()

    def augment_strategy_2_spatial_jitter(
            self,
            traj: np.ndarray,
            ratio: float = 0.3,
            max_distance: int = 5  # 邻居表中的前N个作为"邻近"POI
    ) -> np.ndarray:
        """
        策略2: 空间抖动 - 替换POI为邻近POI
        只替换节点特征,不改变边关系
        """
        traj = traj.copy()
        L = len(traj)
        num_jitter = max(1, int(L * ratio))
        jitter_indices = self.rng.choice(L, size=num_jitter, replace=False)

        for idx in jitter_indices:
            poi = int(traj[idx])
            spatial_neighbors = self.neighbors_spatial[poi]

            if len(spatial_neighbors) > 0:
                # 只从前max_distance个邻居中选择 (距离更近)
                nearby = spatial_neighbors[:min(max_distance, len(spatial_neighbors))]
                traj[idx] = self.rng.choice(nearby)

        return traj

    def augment_strategy_3_temporal_resampling(
            self,
            traj: np.ndarray,
            keep_interval: int = 2
    ) -> np.ndarray:
        """
        策略3: 时间重采样 - 固定间隔降采样
        keep_interval=2 表示每2个点保留1个 (保留50%)
        """
        if len(traj) < 2:
            return traj.copy()

        # 每keep_interval个保留1个
        sampled = traj[::keep_interval]

        # 确保至少有2个点
        if len(sampled) < 2:
            sampled = traj[:2]

        return sampled

    def augment_strategy_4_subsequence(
            self,
            traj: np.ndarray,
            ratio: float = 0.7
    ) -> np.ndarray:
        """
        策略4: 子序列提取 - 连续片段
        ratio=0.7 表示保留70%长度的连续子序列
        """
        L = len(traj)
        if L < 2:
            return traj.copy()

        keep_length = max(2, int(L * ratio))

        # 随机选择起始位置
        if L > keep_length:
            start = self.rng.integers(0, L - keep_length + 1)
            return traj[start:start + keep_length].copy()
        else:
            return traj.copy()

    def augment(
            self,
            traj: np.ndarray,
            strategy: str = 'random'
    ) -> Tuple[np.ndarray, str]:
        """
        应用增强策略

        strategy: 'edge_mask' / 'spatial_jitter' / 'temporal_resample' / 'subsequence' / 'random'
        返回: (增强后的轨迹, 使用的策略名称)
        """
        if strategy == 'random':
            strategy = self.rng.choicea([
                'edge_mask',
                'spatial_jitter',
                'temporal_resample',
                'subsequence'
            ])

        if strategy == 'edge_mask':
            return self.augment_strategy_1_edge_mask(traj), 'edge_mask'
        elif strategy == 'spatial_jitter':
            return self.augment_strategy_2_spatial_jitter(traj, ratio=0.3), 'spatial_jitter'
        elif strategy == 'temporal_resample':
            return self.augment_strategy_3_temporal_resampling(traj, keep_interval=2), 'temporal_resample'
        elif strategy == 'subsequence':
            return self.augment_strategy_4_subsequence(traj, ratio=0.7), 'subsequence'
        else:
            raise ValueError(f"Unknown strategy: {strategy}")


# ==================== Semantic utils ====================

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

            category_lookup[nid] = cat + 1
            subclass_lookup[nid] = sub + 1

    category_vocab = max_cat + 2
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
        "lengths": lengths,
    }


def _move_semantic_pack_to_device(semantic_pack, device):
    out = {}
    for k, v in semantic_pack.items():
        out[k] = v.to(device) if isinstance(v, torch.Tensor) else v
    return out


def _auto_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==================== IO ====================

def load_poi_neighbors(poi_file: str):
    data = np.load(poi_file, allow_pickle=True)
    return data["neighbors"]


def load_traindata(train_file: str):
    data = np.load(train_file, allow_pickle=True)
    x = data["train_list"]
    train_idx = data["train_idx"]
    train_pos = data["train_pos"]
    train_prob = data["train_prob"]
    return x, train_idx, train_pos, train_prob


def load_valdata(val_file: str):
    data = np.load(val_file, allow_pickle=True)
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

    idx_key = None
    for k in ["val_idx", "test_idx"]:
        if k in data:
            idx_key = k
            break
    if idx_key is None:
        raise KeyError(f"Cannot find trajectory list nor val_idx/test_idx in {val_file}")

    idx_global = np.asarray(data[idx_key], dtype=np.int64)
    tra_path = getattr(args, "tra_file", None)
    if tra_path is None:
        tra_path = getattr(args, "raw_tra_file", None)
    if tra_path is None:
        raise KeyError(f"{val_file} is a split file, but args.tra_file is not set")

    tra_all = np.load(tra_path, allow_pickle=True)
    x = tra_all[idx_global]
    y = idx_global
    return x, y


# ==================== Dataset ====================

class MyData(Dataset):
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
    def __init__(self, x, y=None):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        yi = None if self.y is None else self.y[i]
        return self.x[i], yi, i


# ==================== Main loaders ====================

def TrainValueDataLoader(train_file: str, poi_file: str, batchsize: int):
    """
    训练loader (两视图增强版)
    每个batch生成: (anchor, view1, view2)
    每次随机选择一种增强策略
    """
    semantic_lookup = load_semantic_info(args.semantic_file, args.nodes)
    args.category_vocab, args.subclass_vocab = semantic_lookup[2], semantic_lookup[3]

    train_x, train_idx_global, train_pos_global, train_prob = load_traindata(train_file)
    neighbors = load_poi_neighbors(poi_file)
    tra_len = int(train_x.shape[0])
    device = _auto_device()

    global2local = {int(g): int(i) for i, g in enumerate(train_idx_global)}

    MAX_T = int(getattr(args, "max_seq_len", 1024))
    KEEP = "last"

    rng = np.random.default_rng(int(getattr(args, "seed", 123)))

    # 构建图结构
    from collections import defaultdict
    topk_graph = defaultdict(list)
    kg_file = getattr(args, "kg_multi_rel_file", None)
    if kg_file and os.path.exists(kg_file):
        with open(kg_file, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    u, v, r = line.strip().split()
                    u, v, r = int(u), int(v), int(r)
                    topk_graph[u].append((v, r))
                    topk_graph[v].append((u, r))
                except:
                    pass

    # 创建增强器
    augmentor = TrajectoryAugmentor(
        topk_graph=topk_graph,
        neighbors_spatial=neighbors,
        num_nodes=args.nodes,
        seed=args.seed
    )

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

        for _ in range(K):
            jg = int(rng.choice(pos_global_row, p=prob)) if prob is not None else int(rng.choice(pos_global_row))
            if jg == int(anchor_global):
                continue
            jl = global2local.get(jg, None)
            if jl is not None and jl != anchor_local:
                return jl

        for jg in pos_global_row:
            jg = int(jg)
            if jg == int(anchor_global):
                continue
            jl = global2local.get(jg, None)
            if jl is not None and jl != anchor_local:
                return jl

        jl = int(rng.integers(0, tra_len))
        while jl == anchor_local:
            jl = int(rng.integers(0, tra_len))
        return jl

    def collate_fn_neg(data_tuple):
        data_tuple.sort(key=lambda x: len(x[0]), reverse=True)

        # 原始轨迹 (anchor)
        data = [torch.LongTensor(_truncate_seq(sq[0], MAX_T, KEEP)) for sq in data_tuple]
        idx_list = [int(sq[3]) for sq in data_tuple]
        idx_global_list = [int(sq[4]) for sq in data_tuple]

        # 生成两个增强视图 (每次随机选一种策略)
        use_aug = getattr(args, 'use_augmentation', True)

        if use_aug:
            data_view1 = []
            data_view2 = []
            aug_strategies = []

            for sq in data_tuple:
                traj = sq[0]

                # 随机选择一种增强策略
                strategy = rng.choice([
                    'edge_mask',
                    'spatial_jitter',
                    'temporal_resample',
                    'subsequence'
                ])
                aug_strategies.append(strategy)

                # 生成两个增强视图 (同一策略)
                traj_v1, _ = augmentor.augment(traj, strategy=strategy)
                traj_v2, _ = augmentor.augment(traj, strategy=strategy)

                data_view1.append(torch.LongTensor(_truncate_seq(traj_v1, MAX_T, KEEP)))
                data_view2.append(torch.LongTensor(_truncate_seq(traj_v2, MAX_T, KEEP)))
        else:
            data_view1 = [d.clone() for d in data]
            data_view2 = [d.clone() for d in data]
            aug_strategies = ['none'] * len(data)

        # 正样本
        label_indices = []
        for sq in data_tuple:
            pos_local = _sample_pos_local(sq[1], sq[2], int(sq[3]), int(sq[4]))
            label_indices.append(pos_local)

        data_label = [torch.LongTensor(_truncate_seq(train_x[d], MAX_T, KEEP)) for d in label_indices]

        # 负样本
        data_neg = []
        for b in range(len(data)):
            neg = int(rng.integers(0, tra_len))
            while neg == idx_list[b] or neg == label_indices[b]:
                neg = int(rng.integers(0, tra_len))
            data_neg.append(torch.LongTensor(_truncate_seq(train_x[neg], MAX_T, KEEP)))

        # 长度
        data_length = [len(sq) for sq in data]
        neg_length = [len(sq) for sq in data_neg]
        label_length = [len(sq) for sq in data_label]
        view1_length = [len(sq) for sq in data_view1]
        view2_length = [len(sq) for sq in data_view2]

        # POI-level pos/neg
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

        # traj internal
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

        # semantic packs
        semantic_anchor = _build_semantic_pack(data, semantic_lookup[0], semantic_lookup[1])
        semantic_pos = _build_semantic_pack(data_label, semantic_lookup[0], semantic_lookup[1])
        semantic_neg = _build_semantic_pack(data_neg, semantic_lookup[0], semantic_lookup[1])
        semantic_view1 = _build_semantic_pack(data_view1, semantic_lookup[0], semantic_lookup[1])
        semantic_view2 = _build_semantic_pack(data_view2, semantic_lookup[0], semantic_lookup[1])

        # pad sequences
        data = rnn_utils.pad_sequence(data, batch_first=True, padding_value=0)
        data_label = rnn_utils.pad_sequence(data_label, batch_first=True, padding_value=0)
        data_neg = rnn_utils.pad_sequence(data_neg, batch_first=True, padding_value=0)
        data_view1 = rnn_utils.pad_sequence(data_view1, batch_first=True, padding_value=0)
        data_view2 = rnn_utils.pad_sequence(data_view2, batch_first=True, padding_value=0)

        traj_poi_pos_tensor = rnn_utils.pad_sequence(traj_poi_pos, batch_first=True, padding_value=0)
        traj_poi_neg_tensor = rnn_utils.pad_sequence(traj_poi_neg, batch_first=True, padding_value=0)

        poi_pos_tensor = rnn_utils.pad_sequence(poi_pos, batch_first=True, padding_value=0)
        poi_neg_tensor = rnn_utils.pad_sequence(poi_neg, batch_first=True, padding_value=0)

        # move to device
        data = data.to(device)
        data_label = data_label.to(device)
        data_neg = data_neg.to(device)
        data_view1 = data_view1.to(device)
        data_view2 = data_view2.to(device)

        traj_poi_pos_tensor = traj_poi_pos_tensor.to(device)
        traj_poi_neg_tensor = traj_poi_neg_tensor.to(device)
        poi_pos_tensor = poi_pos_tensor.to(device)
        poi_neg_tensor = poi_neg_tensor.to(device)

        semantic_anchor_dev = _move_semantic_pack_to_device(semantic_anchor, device)
        semantic_pos_dev = _move_semantic_pack_to_device(semantic_pos, device)
        semantic_neg_dev = _move_semantic_pack_to_device(semantic_neg, device)
        semantic_view1_dev = _move_semantic_pack_to_device(semantic_view1, device)
        semantic_view2_dev = _move_semantic_pack_to_device(semantic_view2, device)

        return (
            data, data_neg, data_label,
            data_length, neg_length, label_length,
            traj_poi_pos_tensor, traj_poi_neg_tensor,
            poi_pos_tensor, poi_neg_tensor,
            semantic_anchor_dev, semantic_pos_dev, semantic_neg_dev,
            data_view1, data_view2,
            view1_length, view2_length,
            semantic_view1_dev, semantic_view2_dev,
            aug_strategies,  # 记录使用的增强策略
        )

    dataset = DataLoader(
        MyData(train_x, train_pos_global, train_prob, train_idx_global),
        batch_size=batchsize,
        shuffle=True,
        collate_fn=collate_fn_neg,
    )
    return dataset


def TrainDataValLoader(train_file: str, batchsize: int):
    semantic_lookup = load_semantic_info(args.semantic_file, args.nodes)
    args.category_vocab, args.subclass_vocab = semantic_lookup[2], semantic_lookup[3]
    device = _auto_device()

    train_x, train_idx_global, _, _ = load_traindata(train_file)

    MAX_T = int(getattr(args, "max_seq_len", 1024))
    KEEP = "last"

    dummy_pos = np.zeros((len(train_x), 1), dtype=np.int64)
    dummy_prob = np.zeros((len(train_x), 1), dtype=np.float32)

    def collate_fn_eval(data_tuple):
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
    semantic_lookup = load_semantic_info(args.semantic_file, args.nodes)
    args.category_vocab, args.subclass_vocab = semantic_lookup[2], semantic_lookup[3]
    device = _auto_device()

    val_x, val_y = load_valdata(val_file)

    MAX_T = int(getattr(args, "max_seq_len", 1024))
    KEEP = "last"

    def collate_fn_eval(batch):
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
    return ValValueDataLoader(test_file, batchsize)