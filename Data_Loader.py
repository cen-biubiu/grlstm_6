from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import Dataset, DataLoader

from pars_args import args

# 导入增强器
import sys
import os
sys.path.append(os.path.dirname(__file__))

# semantic cache
_SEMANTIC_CACHE: Dict[Tuple[str, int], Tuple[np.ndarray, np.ndarray, int, int]] = {}


# ==================== 增强器定义 ====================

class TrajectoryAugmentor:
    """轻量级增强器 (核心功能)"""
    
    def __init__(
        self,
        topk_graph: Dict[int, List[Tuple[int, int]]],
        category_lookup: np.ndarray,
        neighbors_spatial: np.ndarray,
        num_nodes: int,
        seed: int = 42
    ):
        self.topk_graph = topk_graph
        self.category_lookup = category_lookup
        self.neighbors_spatial = neighbors_spatial
        self.num_nodes = num_nodes
        self.rng = np.random.default_rng(seed)
    
    def neighbor_substitution(self, traj: np.ndarray, ratio: float = 0.3) -> np.ndarray:
        """邻居替换"""
        traj = traj.copy()
        L = len(traj)
        num_replace = max(1, int(L * ratio))
        replace_indices = self.rng.choice(L, size=num_replace, replace=False)
        
        for idx in replace_indices:
            poi = int(traj[idx])
            neighbors = self.topk_graph.get(poi, [])
            if neighbors:
                new_poi, _ = neighbors[self.rng.integers(0, len(neighbors))]
                traj[idx] = new_poi
        return traj
    
    def poi_generalization(self, traj: np.ndarray, ratio: float = 0.2) -> np.ndarray:
        """POI泛化"""
        traj = traj.copy()
        L = len(traj)
        num_replace = max(1, int(L * ratio))
        replace_indices = self.rng.choice(L, size=num_replace, replace=False)
        
        for idx in replace_indices:
            poi = int(traj[idx])
            category = self.category_lookup[poi]
            same_category = np.where(self.category_lookup == category)[0]
            
            if len(same_category) > 1:
                candidates = same_category[same_category != poi]
                if len(candidates) > 0:
                    traj[idx] = self.rng.choice(candidates)
        return traj
    
    def spatial_jittering(self, traj: np.ndarray, ratio: float = 0.25) -> np.ndarray:
        """空间抖动"""
        traj = traj.copy()
        L = len(traj)
        num_jitter = max(1, int(L * ratio))
        jitter_indices = self.rng.choice(L, size=num_jitter, replace=False)
        
        for idx in jitter_indices:
            poi = int(traj[idx])
            spatial_neighbors = self.neighbors_spatial[poi]
            if len(spatial_neighbors) > 0:
                traj[idx] = self.rng.choice(spatial_neighbors)
        return traj
    
    def temporal_resampling(
        self, 
        traj: np.ndarray, 
        scale_range: Tuple[float, float] = (0.7, 1.3)
    ) -> np.ndarray:
        """时间重采样"""
        L = len(traj)
        scale = self.rng.uniform(*scale_range)
        new_length = max(2, int(L * scale))
        new_indices = np.linspace(0, L - 1, new_length)
        resampled = [traj[int(np.round(idx))] for idx in new_indices]
        return np.array(resampled)
    
    def subsequence_sampling(self, traj: np.ndarray, ratio: float = 0.7) -> np.ndarray:
        """子序列提取"""
        L = len(traj)
        keep_length = max(2, int(L * ratio))
        start = self.rng.integers(0, L - keep_length + 1)
        return traj[start:start + keep_length].copy()
    
    def order_perturbation(self, traj: np.ndarray, window_size: int = 3) -> np.ndarray:
        """顺序扰动"""
        traj = traj.copy()
        L = len(traj)
        if L < window_size:
            return traj
        start = self.rng.integers(0, L - window_size + 1)
        window = traj[start:start + window_size].copy()
        self.rng.shuffle(window)
        traj[start:start + window_size] = window
        return traj
    
    def augment_spatial(self, traj: np.ndarray, strategy: str = 'random') -> np.ndarray:
        """空间增强 (随机选择策略)"""
        if strategy == 'random':
            strategy = self.rng.choice(['neighbor', 'generalize', 'jitter'])
        
        if strategy == 'neighbor':
            return self.neighbor_substitution(traj, ratio=0.3)
        elif strategy == 'generalize':
            return self.poi_generalization(traj, ratio=0.2)
        elif strategy == 'jitter':
            return self.spatial_jittering(traj, ratio=0.25)
        return traj.copy()
    
    def augment_temporal(self, traj: np.ndarray, strategy: str = 'random') -> np.ndarray:
        """时间增强 (随机选择策略)"""
        if strategy == 'random':
            strategy = self.rng.choice(['resample', 'subsequence', 'perturb'])
        
        if strategy == 'resample':
            return self.temporal_resampling(traj)
        elif strategy == 'subsequence':
            return self.subsequence_sampling(traj, ratio=0.7)
        elif strategy == 'perturb':
            return self.order_perturbation(traj, window_size=3)
        return traj.copy()


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
    🔥 训练loader (增强版)
    
    新增功能:
    - 空间增强轨迹
    - 时间增强轨迹
    - 返回三元组: (anchor, spatial_aug, temporal_aug)
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
    
    # 🔥 构建图结构 (用于增强)
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
    
    # 🔥 创建增强器
    augmentor = TrajectoryAugmentor(
        topk_graph=topk_graph,
        category_lookup=semantic_lookup[0],
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

        # 🔥 原始轨迹
        data = [torch.LongTensor(_truncate_seq(sq[0], MAX_T, KEEP)) for sq in data_tuple]
        idx_list = [int(sq[3]) for sq in data_tuple]
        idx_global_list = [int(sq[4]) for sq in data_tuple]

        # 🔥 增强轨迹
        use_aug = getattr(args, 'use_augmentation', True)
        if use_aug:
            data_spatial_aug = []
            data_temporal_aug = []
            
            for sq in data_tuple:
                traj = sq[0]
                # 空间增强
                traj_spa = augmentor.augment_spatial(traj)
                data_spatial_aug.append(torch.LongTensor(_truncate_seq(traj_spa, MAX_T, KEEP)))
                # 时间增强
                traj_temp = augmentor.augment_temporal(traj)
                data_temporal_aug.append(torch.LongTensor(_truncate_seq(traj_temp, MAX_T, KEEP)))
        else:
            # 不使用增强
            data_spatial_aug = [d.clone() for d in data]
            data_temporal_aug = [d.clone() for d in data]

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
        spatial_aug_length = [len(sq) for sq in data_spatial_aug]
        temporal_aug_length = [len(sq) for sq in data_temporal_aug]

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
        semantic_spatial_aug = _build_semantic_pack(data_spatial_aug, semantic_lookup[0], semantic_lookup[1])
        semantic_temporal_aug = _build_semantic_pack(data_temporal_aug, semantic_lookup[0], semantic_lookup[1])

        # pad sequences
        data = rnn_utils.pad_sequence(data, batch_first=True, padding_value=0)
        data_label = rnn_utils.pad_sequence(data_label, batch_first=True, padding_value=0)
        data_neg = rnn_utils.pad_sequence(data_neg, batch_first=True, padding_value=0)
        data_spatial_aug = rnn_utils.pad_sequence(data_spatial_aug, batch_first=True, padding_value=0)
        data_temporal_aug = rnn_utils.pad_sequence(data_temporal_aug, batch_first=True, padding_value=0)

        traj_poi_pos_tensor = rnn_utils.pad_sequence(traj_poi_pos, batch_first=True, padding_value=0)
        traj_poi_neg_tensor = rnn_utils.pad_sequence(traj_poi_neg, batch_first=True, padding_value=0)

        poi_pos_tensor = rnn_utils.pad_sequence(poi_pos, batch_first=True, padding_value=0)
        poi_neg_tensor = rnn_utils.pad_sequence(poi_neg, batch_first=True, padding_value=0)

        # move to device
        data = data.to(device)
        data_label = data_label.to(device)
        data_neg = data_neg.to(device)
        data_spatial_aug = data_spatial_aug.to(device)
        data_temporal_aug = data_temporal_aug.to(device)

        traj_poi_pos_tensor = traj_poi_pos_tensor.to(device)
        traj_poi_neg_tensor = traj_poi_neg_tensor.to(device)
        poi_pos_tensor = poi_pos_tensor.to(device)
        poi_neg_tensor = poi_neg_tensor.to(device)

        semantic_anchor_dev = _move_semantic_pack_to_device(semantic_anchor, device)
        semantic_pos_dev = _move_semantic_pack_to_device(semantic_pos, device)
        semantic_neg_dev = _move_semantic_pack_to_device(semantic_neg, device)
        semantic_spatial_aug_dev = _move_semantic_pack_to_device(semantic_spatial_aug, device)
        semantic_temporal_aug_dev = _move_semantic_pack_to_device(semantic_temporal_aug, device)

        return (
            data, data_neg, data_label,
            data_length, neg_length, label_length,
            traj_poi_pos_tensor, traj_poi_neg_tensor,
            poi_pos_tensor, poi_neg_tensor,
            semantic_anchor_dev, semantic_pos_dev, semantic_neg_dev,
            data_spatial_aug, data_temporal_aug,
            spatial_aug_length, temporal_aug_length,
            semantic_spatial_aug_dev, semantic_temporal_aug_dev,
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