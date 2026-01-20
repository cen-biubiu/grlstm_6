import logging
import os
from collections import defaultdict
from typing import Dict, Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from Data_Loader import load_semantic_info


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 20000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        time_steps = x.size(1)
        if time_steps > self.pe.size(1):
            self._expand_pe(time_steps, x.device)
        return x + self.pe[:, :time_steps, :]

    def _expand_pe(self, new_len: int, device):
        d_model = self.pe.size(2)
        pe = torch.zeros(new_len, d_model, device=device)
        position = torch.arange(0, new_len, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, device=device).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)


class GRLSTM(nn.Module):
    """
    Dual-branch encoder with CONSISTENT pooling strategy:
      - Spatial branch: KG+road neighbors (relation-biased aggregation)
      - Semantic branch: category/subclass Transformer
      - Fusion: dual cross-attn + FIXED weights (ALPHA/DELTA)
      - Head: FFN (residual+LN)
      - Pool: masked AvgPool for ALL representations
    """

    def __init__(self, args, device, batch_first=True):
        super().__init__()
        self.nodes = args.nodes
        self.latent_dim = args.latent_dim
        self.device = device
        self.batch_first = batch_first
        self.num_heads = args.num_heads
        self.topk_neighbors = args.topk_neighbors
        self.self_loop_rel = 9
        self.rel_vocab = self.self_loop_rel + 1

        logging.info("Initializing model: latent_dim=%d", self.latent_dim)

        # Fixed fusion weights
        alpha_spa = float(getattr(args, "alpha_spa", 0.3))
        delta_sem = float(getattr(args, "delta_sem", 0.5))
        s = max(alpha_spa + delta_sem, 1e-8)
        w_spa = alpha_spa / s
        w_sem = delta_sem / s

        self.register_buffer("w_sem", torch.tensor(w_sem, dtype=torch.float32))
        self.register_buffer("w_spa", torch.tensor(w_spa, dtype=torch.float32))
        logging.info("Fixed fusion weights: w_sem=%.4f, w_spa=%.4f", w_sem, w_spa)

        # Semantic meta
        category_lookup, subclass_lookup, cat_vocab, sub_vocab = load_semantic_info(
            args.semantic_file, args.nodes
        )
        args.category_vocab = getattr(args, "category_vocab", cat_vocab)
        args.subclass_vocab = getattr(args, "subclass_vocab", sub_vocab)
        self.register_buffer("category_lookup", torch.as_tensor(category_lookup, dtype=torch.long))
        self.register_buffer("subclass_lookup", torch.as_tensor(subclass_lookup, dtype=torch.long))

        # Spatial branch
        neighbors_np = np.load(args.poi_file, allow_pickle=True)["neighbors"]
        self.topk_graph = self._build_topk_graph(args.kg_multi_rel_file, neighbors_np)

        poi_features = np.load(args.poi_feature_file)
        self.node_embedding = nn.Embedding.from_pretrained(
            torch.tensor(poi_features, dtype=torch.float32),
            freeze=False,
        )
        # padding_idx=None 没必要写；但保留你原意
        self.rel_embedding = nn.Embedding(self.rel_vocab, self.latent_dim)

        # 🔥 预计算 rel_bias（修复 device mismatch 的关键在这里）
        self._precompute_rel_bias()

        self.spatial_pos_enc = PositionalEncoding(self.latent_dim, max_len=20000)

        # Semantic branch
        self.token_embedding = nn.Embedding(args.subclass_vocab, self.latent_dim, padding_idx=0)
        self.segment_embedding = nn.Embedding(args.category_vocab, self.latent_dim, padding_idx=0)
        self.pos_embedding = nn.Embedding(args.max_seq_len, self.latent_dim)
        self.max_seq_len = args.max_seq_len

        sem_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.latent_dim,
            nhead=args.num_heads,
            dim_feedforward=args.trans_ffn_dim,
            dropout=args.trans_dropout,
            batch_first=True,
        )
        self.semantic_encoder = nn.TransformerEncoder(
            sem_encoder_layer, num_layers=args.trans_layers
        )

        # Dual cross-attn
        self.co_attn_sem = nn.MultiheadAttention(
            embed_dim=self.latent_dim,
            num_heads=getattr(args, "co_heads", args.num_heads),
            batch_first=True,
            dropout=args.trans_dropout,
        )
        self.co_attn_spa = nn.MultiheadAttention(
            embed_dim=self.latent_dim,
            num_heads=getattr(args, "co_heads", args.num_heads),
            batch_first=True,
            dropout=args.trans_dropout,
        )

        self.fuse_linear = nn.Linear(self.latent_dim * 2, self.latent_dim)
        self.fuse_norm = nn.LayerNorm(self.latent_dim)
        self.fuse_dropout = nn.Dropout(args.trans_dropout)

        # FFN block
        self.fusion_ffn = nn.Sequential(
            nn.Linear(self.latent_dim, args.trans_ffn_dim),
            nn.GELU(),
            nn.Dropout(args.trans_dropout),
            nn.Linear(args.trans_ffn_dim, self.latent_dim),
            nn.Dropout(args.trans_dropout),
        )
        self.fusion_ffn_norm = nn.LayerNorm(self.latent_dim)

    def _build_topk_graph(self, kg_multi_rel_file: str, neighbors_np: np.ndarray):
        graph: Dict[int, List[Tuple[int, int]]] = defaultdict(list)

        if os.path.exists(kg_multi_rel_file):
            with open(kg_multi_rel_file, "r") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        u, v, r = line.strip().split()
                        u, v, r = int(u), int(v), int(r)
                    except ValueError:
                        continue
                    graph[u].append((v, r))
                    graph[v].append((u, r))

        for nid in range(self.nodes):
            candidates = graph.get(nid, [])
            for nb in neighbors_np[nid]:
                candidates.append((int(nb), 0))

            seen = set()
            top = []
            for nb, rel in candidates:
                if nb in seen:
                    continue
                seen.add(nb)
                top.append((nb, rel))
                if len(top) >= self.topk_neighbors:
                    break
            graph[nid] = top
        return graph

    # ✅ 更稳：mask 构造跟随输入 device，不依赖 self.device 字段
    def _build_padding_mask(self, lengths, max_len, device: Optional[torch.device] = None):
        dev = device if device is not None else self.rel_bias_cache.device
        lens = torch.as_tensor(lengths, device=dev)
        return torch.arange(max_len, device=dev).unsqueeze(0) >= lens.unsqueeze(1)

    def _masked_avg_pool(self, seq: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        valid = (~padding_mask).float().unsqueeze(-1)
        seq = seq * valid
        denom = valid.sum(dim=1).clamp(min=1.0)
        return seq.sum(dim=1) / denom

    def _precompute_rel_bias(self):
        """
        🔥 优化1：预计算所有节点的关系偏置（只在初始化时执行一次）
        ✅ 修复：切断 autograd（no_grad + detach），避免 backward 时 CPU/CUDA 混用
        """
        logging.info("[Model] Precomputing rel_bias for %d nodes...", self.nodes)

        # cache 放 CPU，节省显存；buffer 会在 model.to(device) 时自动迁移
        rel_bias = torch.zeros((self.nodes, self.latent_dim), dtype=torch.float32)

        with torch.no_grad():
            # 关键：detach，断开计算图；并在 CPU 上取 weight
            rel_w = self.rel_embedding.weight.detach().cpu()  # [rel_vocab, D]

            for nid in range(self.nodes):
                neighbors = self.topk_graph.get(nid, [])
                if not neighbors:
                    continue

                rel_ids = [rel for _, rel in neighbors]
                if len(rel_ids) > 0:
                    rel_ids_t = torch.tensor(rel_ids, dtype=torch.long)  # CPU tensor
                    rel_bias[nid] = rel_w[rel_ids_t].mean(dim=0)

        # 注册为 buffer：常量，不参与梯度
        self.register_buffer("rel_bias_cache", rel_bias, persistent=True)

        # 安全断言：必须不需要梯度
        assert not self.rel_bias_cache.requires_grad, "rel_bias_cache should be a constant buffer (requires_grad=False)."

        logging.info("[Model] Precomputed rel_bias: shape=%s", rel_bias.shape)

    def _encode_spatial(
        self,
        nodes: torch.Tensor,
        lengths,
        poi: Optional[torch.Tensor] = None,
        traj_poi: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        poi_lengths: Optional[List[int]] = None,
        traj_poi_lengths: Optional[List[int]] = None,
    ):
        dev = nodes.device

        if mask is None:
            mask = self._build_padding_mask(lengths, nodes.size(1), device=dev)

        valid = nodes.masked_select(~mask)

        if poi is not None:
            if poi_lengths is not None:
                poi_mask = self._build_padding_mask(poi_lengths, poi.size(1), device=poi.device)
            else:
                poi_mask = torch.zeros_like(poi, dtype=torch.bool, device=poi.device)
            valid = torch.cat([valid, poi.masked_select(~poi_mask)], dim=0)

        if traj_poi is not None:
            if traj_poi_lengths is not None:
                traj_mask = self._build_padding_mask(traj_poi_lengths, traj_poi.size(1), device=traj_poi.device)
            else:
                traj_mask = torch.zeros_like(traj_poi, dtype=torch.bool, device=traj_poi.device)
            valid = torch.cat([valid, traj_poi.masked_select(~traj_mask)], dim=0)

        unique_nodes = torch.unique(valid)
        if unique_nodes.numel() == 0:
            unique_nodes = torch.tensor([0], device=dev, dtype=torch.long)

        edges: List[Tuple[int, int, int]] = []
        for nid in unique_nodes.tolist():
            edges.append((nid, nid, self.self_loop_rel))
            for nb, rel in self.topk_graph.get(nid, []):
                edges.append((nb, nid, rel))

        if not edges:
            edges = [(0, 0, self.self_loop_rel)]

        src = torch.as_tensor([e[0] for e in edges], device=dev, dtype=torch.long)
        dst = torch.as_tensor([e[1] for e in edges], device=dev, dtype=torch.long)
        rel = torch.as_tensor([e[2] for e in edges], device=dev, dtype=torch.long)

        # ✅ 关键：rel_bias_cache 显式搬到 dev（避免偶发不一致）
        node_feats = self.node_embedding.weight.to(dev) + self.rel_bias_cache.to(dev)

        rel_emb = self.rel_embedding(rel)  # rel 在 dev 上，所以 rel_emb 在 dev 上
        messages = node_feats[src] + rel_emb

        agg = torch.zeros_like(node_feats)
        agg.index_add_(0, dst, messages)

        deg = torch.zeros(self.nodes, device=dev)
        deg.index_add_(0, dst, torch.ones(len(dst), device=dev))
        deg = deg.clamp(min=1).unsqueeze(-1)
        agg = agg / deg

        spatial_features = agg + node_feats
        spatial_features = F.normalize(spatial_features, p=2, dim=-1) * np.sqrt(self.latent_dim)

        spatial_seq = spatial_features[nodes]

        poi_seq = spatial_features[poi] if poi is not None else None
        traj_poi_seq = spatial_features[traj_poi] if traj_poi is not None else None

        return spatial_seq, poi_seq, traj_poi_seq

    def _semantic_from_nodes(self, nodes: torch.Tensor, lengths):
        positions = torch.arange(nodes.size(1), device=nodes.device).unsqueeze(0).expand(nodes.size(0), -1)
        return {
            "tokens": self.subclass_lookup[nodes],
            "segments": self.category_lookup[nodes],
            "positions": positions,
            "padding_mask": self._build_padding_mask(lengths, nodes.size(1), device=nodes.device),
            "lengths": lengths,
        }

    def encode_poi_batch(self, poi_batch: torch.Tensor, poi_lengths):
        """
        🔥 新增方法：单独编码POI序列（用于对比学习）
        避免重复forward整个轨迹
        """
        poi_batch = poi_batch.to(self.device)
        mask = self._build_padding_mask(poi_lengths, poi_batch.size(1), device=poi_batch.device)

        spatial_seq, _, _ = self._encode_spatial(
            poi_batch,
            poi_lengths,
            poi=None,
            traj_poi=None,
            mask=mask
        )

        spatial_seq = self.spatial_pos_enc(spatial_seq)
        return self._masked_avg_pool(spatial_seq, mask)

    def forward(
        self,
        batch_nodes: torch.Tensor,
        batch_lengths,
        semantic: Optional[Dict[str, torch.Tensor]] = None,
        poi: Optional[torch.Tensor] = None,
        traj_poi: Optional[torch.Tensor] = None,
        poi_lengths: Optional[List[int]] = None,
        traj_poi_lengths: Optional[List[int]] = None,
    ):
        batch_nodes = batch_nodes.to(self.device)
        mask = self._build_padding_mask(batch_lengths, batch_nodes.size(1), device=batch_nodes.device)

        spatial_seq, poi_seq, traj_poi_seq = self._encode_spatial(
            batch_nodes,
            batch_lengths,
            poi,
            traj_poi,
            mask,
            poi_lengths=poi_lengths,
            traj_poi_lengths=traj_poi_lengths,
        )
        spatial_seq = self.spatial_pos_enc(spatial_seq)

        if semantic is None:
            semantic = self._semantic_from_nodes(batch_nodes, batch_lengths)

        tokens = semantic["tokens"].to(batch_nodes.device)
        segments = semantic["segments"].to(batch_nodes.device)
        positions = semantic["positions"].to(batch_nodes.device)
        sem_mask = semantic.get("padding_mask", mask).to(batch_nodes.device)

        positions = positions.clamp(max=self.pos_embedding.num_embeddings - 1)

        sem_input = self.token_embedding(tokens) + self.segment_embedding(segments)
        sem_input = sem_input + self.pos_embedding(positions)

        semantic_out = self.semantic_encoder(sem_input, src_key_padding_mask=sem_mask)

        sem_attn, _ = self.co_attn_sem(
            semantic_out, spatial_seq, spatial_seq,
            key_padding_mask=mask,
        )
        spa_attn, _ = self.co_attn_spa(
            spatial_seq, semantic_out, semantic_out,
            key_padding_mask=sem_mask,
        )

        combined = torch.cat([sem_attn, spa_attn], dim=-1)
        weighted = self.w_sem * sem_attn + self.w_spa * spa_attn

        fused = self.fuse_linear(combined)
        fused = self.fuse_norm(fused + self.fuse_dropout(weighted))

        ffn_out = self.fusion_ffn(fused)
        fused = self.fusion_ffn_norm(fused + ffn_out)

        traj_repr = self._masked_avg_pool(fused, mask)

        poi_emb_pooled = None
        traj_poi_emb_pooled = None

        if poi_seq is not None:
            if poi_lengths is None:
                poi_mask = mask
            else:
                poi_mask = self._build_padding_mask(poi_lengths, poi_seq.size(1), device=poi_seq.device)
            poi_emb_pooled = self._masked_avg_pool(poi_seq, poi_mask)

        if traj_poi_seq is not None:
            if traj_poi_lengths is None:
                traj_poi_mask = mask
            else:
                traj_poi_mask = self._build_padding_mask(traj_poi_lengths, traj_poi_seq.size(1), device=traj_poi_seq.device)
            traj_poi_emb_pooled = self._masked_avg_pool(traj_poi_seq, traj_poi_mask)

        return {
            "traj_repr": traj_repr,
            "poi_emb": poi_emb_pooled,
            "traj_poi_emb": traj_poi_emb_pooled,
            "fused_seq": fused,
            "mask": mask,
        }