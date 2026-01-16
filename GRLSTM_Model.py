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

        # Fixed fusion weights (aligned with ground-truth)
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
        self.rel_embedding = nn.Embedding(self.rel_vocab, self.latent_dim, padding_idx=None)
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

    def _build_padding_mask(self, lengths, max_len):
        lens = torch.as_tensor(lengths, device=self.device)
        return torch.arange(max_len, device=self.device).unsqueeze(0) >= lens.unsqueeze(1)

    def _masked_avg_pool(self, seq: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        """
        seq: [B, T, D]
        padding_mask: [B, T] True means padding
        return: [B, D]
        """
        valid = (~padding_mask).float().unsqueeze(-1)
        seq = seq * valid
        denom = valid.sum(dim=1).clamp(min=1.0)
        return seq.sum(dim=1) / denom

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
        if mask is None:
            mask = self._build_padding_mask(lengths, nodes.size(1))

        valid = nodes.masked_select(~mask)

        if poi is not None:
            if poi_lengths is not None:
                poi_mask = self._build_padding_mask(poi_lengths, poi.size(1))
            else:
                poi_mask = torch.zeros_like(poi, dtype=torch.bool)
            valid = torch.cat([valid, poi.masked_select(~poi_mask)], dim=0)

        if traj_poi is not None:
            if traj_poi_lengths is not None:
                traj_mask = self._build_padding_mask(traj_poi_lengths, traj_poi.size(1))
            else:
                traj_mask = torch.zeros_like(traj_poi, dtype=torch.bool)
            valid = torch.cat([valid, traj_poi.masked_select(~traj_mask)], dim=0)

        unique_nodes = torch.unique(valid)
        if unique_nodes.numel() == 0:
            unique_nodes = torch.tensor([0], device=self.device)

        edges: List[Tuple[int, int, int]] = []
        node_rel_sum = {}

        for nid in unique_nodes.tolist():
            edges.append((nid, nid, self.self_loop_rel))

            rel_list = []
            for nb, rel in self.topk_graph.get(nid, []):
                edges.append((nb, nid, rel))
                rel_list.append(rel)
            if rel_list:
                node_rel_sum[nid] = rel_list

        if not edges:
            edges = [(0, 0, self.self_loop_rel)]

        src = torch.as_tensor([e[0] for e in edges], device=self.device, dtype=torch.long)
        dst = torch.as_tensor([e[1] for e in edges], device=self.device, dtype=torch.long)
        rel = torch.as_tensor([e[2] for e in edges], device=self.device, dtype=torch.long)

        rel_bias = torch.zeros((self.nodes, self.latent_dim), device=self.device)
        for nid, rel_ids in node_rel_sum.items():
            rel_embs = self.rel_embedding(torch.tensor(rel_ids, device=self.device))
            rel_bias[nid] = rel_embs.mean(dim=0)

        node_feats = self.node_embedding.weight.to(self.device) + rel_bias
        rel_emb = self.rel_embedding(rel)
        messages = node_feats[src] + rel_emb

        agg = torch.zeros_like(node_feats)
        agg.index_add_(0, dst, messages)

        deg = torch.zeros(self.nodes, device=self.device)
        deg.index_add_(0, dst, torch.ones(len(dst), device=self.device))
        deg = deg.clamp(min=1).unsqueeze(-1)
        agg = agg / deg

        spatial_features = agg + node_feats
        spatial_features = F.normalize(spatial_features, p=2, dim=-1) * np.sqrt(self.latent_dim)

        spatial_seq = spatial_features[nodes]
        
        # Return sequence representations for POI-level contrast
        poi_seq = spatial_features[poi] if poi is not None else None
        traj_poi_seq = spatial_features[traj_poi] if traj_poi is not None else None
        
        return spatial_seq, poi_seq, traj_poi_seq

    def _semantic_from_nodes(self, nodes: torch.Tensor, lengths):
        positions = torch.arange(nodes.size(1), device=nodes.device).unsqueeze(0).expand(nodes.size(0), -1)
        return {
            "tokens": self.subclass_lookup[nodes],
            "segments": self.category_lookup[nodes],
            "positions": positions,
            "padding_mask": self._build_padding_mask(lengths, nodes.size(1)),
            "lengths": lengths,
        }

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
        mask = self._build_padding_mask(batch_lengths, batch_nodes.size(1))

        # Spatial encoding
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

        # Semantic encoding
        if semantic is None:
            semantic = self._semantic_from_nodes(batch_nodes, batch_lengths)

        tokens = semantic["tokens"].to(self.device)
        segments = semantic["segments"].to(self.device)
        positions = semantic["positions"].to(self.device)
        sem_mask = semantic.get("padding_mask", mask).to(self.device)

        positions = positions.clamp(max=self.pos_embedding.num_embeddings - 1)

        sem_input = self.token_embedding(tokens) + self.segment_embedding(segments)
        sem_input = sem_input + self.pos_embedding(positions)

        semantic_out = self.semantic_encoder(sem_input, src_key_padding_mask=sem_mask)

        # Dual cross-attention
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

        # FFN block
        ffn_out = self.fusion_ffn(fused)
        fused = self.fusion_ffn_norm(fused + ffn_out)

        # CRITICAL FIX: Use masked average pooling for trajectory representation
        traj_repr = self._masked_avg_pool(fused, mask)
        
        # Also pool POI-level sequences for loss computation
        poi_emb_pooled = None
        traj_poi_emb_pooled = None
        
        if poi_seq is not None:
            poi_mask = self._build_padding_mask(poi_lengths, poi_seq.size(1)) if poi_lengths else mask
            poi_emb_pooled = self._masked_avg_pool(poi_seq, poi_mask)
            
        if traj_poi_seq is not None:
            traj_poi_mask = self._build_padding_mask(traj_poi_lengths, traj_poi_seq.size(1)) if traj_poi_lengths else mask
            traj_poi_emb_pooled = self._masked_avg_pool(traj_poi_seq, traj_poi_mask)

        return {
            "traj_repr": traj_repr,  # [B, D] - pooled
            "poi_emb": poi_emb_pooled,  # [B, D] - pooled
            "traj_poi_emb": traj_poi_emb_pooled,  # [B, D] - pooled
            "fused_seq": fused,  # [B, T, D] - sequence (for future use)
            "mask": mask,
        }
