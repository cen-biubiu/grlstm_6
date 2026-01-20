# Trainer.py
import logging
import os
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm


class Trainer:
    def __init__(
            self,
            model,
            train_data_loader,
            val_data_loader,
            n_epochs,
            lr,
            save_epoch_int,
            model_folder,
            device,
            grad_accum_steps: int = 1,
    ):
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.n_epochs = n_epochs
        self.lr = lr
        self.save_epoch_int = save_epoch_int
        self.model_folder = model_folder
        self.device = device
        self.model = model.to(self.device)
        self.grad_accum_steps = max(1, grad_accum_steps)

        if not os.path.exists(model_folder):
            os.makedirs(model_folder)

        self.optim = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr,
        )

    # -----------------------------
    # helpers: seq -> fixed (兼容 [B,T,D] / [B,D])
    # -----------------------------
    @staticmethod
    def _masked_avg_pool(seq: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        """
        seq: [B, T, D]
        padding_mask: [B, T], True means padding
        return: [B, D]
        """
        valid = (~padding_mask).float().unsqueeze(-1)  # [B, T, 1]
        seq = seq * valid
        denom = valid.sum(dim=1).clamp(min=1.0)  # [B, 1]
        return seq.sum(dim=1) / denom  # [B, D]

    @staticmethod
    def _to_fixed(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x could be:
          - [B, D]  -> return as-is
          - [B, T, D] -> masked avg pool
          - None -> None
        """
        if x is None:
            return None
        if x.dim() == 2:
            return x
        if x.dim() == 3:
            return Trainer._masked_avg_pool(x, mask)
        raise ValueError(f"Expected x dim=2 or 3, got {x.shape}")

    @staticmethod
    def _dot_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        a,b: [B, D]
        return: [B]
        """
        if a is None or b is None:
            return None
        if a.dim() != 2 or b.dim() != 2:
            raise ValueError(f"Expected [B,D], got a={a.shape}, b={b.shape}")
        return (a * b).sum(dim=-1)

    @staticmethod
    def _bpr_loss(pos_score: torch.Tensor, neg_score: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        BPR: -log(sigmoid(pos-neg))
        """
        if pos_score is None or neg_score is None:
            return None
        sig = (pos_score - neg_score).sigmoid().clamp(min=eps)
        return -(sig.log()).mean()

    def _pass(self, data, train=True):
        (
            batch_x, batch_n, batch_y,
            batch_x_len, batch_n_len, batch_y_len,
            batch_traj_poi_pos, batch_traj_poi_neg,
            poi_pos, poi_neg,
            semantic_anchor, semantic_pos, semantic_neg
        ) = data

        # -------- move tensors --------
        batch_x = batch_x.to(self.device)
        batch_y = batch_y.to(self.device)
        batch_n = batch_n.to(self.device)

        poi_pos = poi_pos.to(self.device)
        poi_neg = poi_neg.to(self.device)
        batch_traj_poi_pos = batch_traj_poi_pos.to(self.device)
        batch_traj_poi_neg = batch_traj_poi_neg.to(self.device)

        # -------- move semantic packs --------
        semantic_anchor = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in semantic_anchor.items()}
        semantic_pos = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in semantic_pos.items()}
        semantic_neg = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in semantic_neg.items()}

        eps = 1e-8

        # ==========================================================
        # 1) traj-level: anchor vs pos/neg  (固定向量对比) ✅
        # ==========================================================
        anchor_out = self.model(batch_x, batch_x_len, semantic_anchor)
        pos_out = self.model(batch_y, batch_y_len, semantic_pos)
        neg_out = self.model(batch_n, batch_n_len, semantic_neg)

        pos_score_1 = self._dot_sim(anchor_out["traj_repr"], pos_out["traj_repr"])
        neg_score_1 = self._dot_sim(anchor_out["traj_repr"], neg_out["traj_repr"])
        loss1 = self._bpr_loss(pos_score_1, neg_score_1, eps=eps)

        if loss1 is None:
            # 理论上不会发生
            loss1 = torch.tensor(0.0, device=self.device)

        # ==========================================================
        # 2) poi-level / traj-ctx-level:
        #    关键修复：poi_pos/poi_neg、traj_poi_pos/neg 是按 anchor 轨迹生成的
        #    所以 loss2/loss3 只围绕 anchor 来做对比（不再用 pos_out/neg_out 的序列）
        # ==========================================================
        loss2 = torch.tensor(0.0, device=self.device)
        loss3 = torch.tensor(0.0, device=self.device)

        # ---- loss2: anchor_traj_repr vs pooled(poi_pos) / pooled(poi_neg) ----
        try:
            # 用同一个 anchor batch_x，但分别喂 poi_pos / poi_neg
            out_poi_pos = self.model(
                batch_x, batch_x_len, semantic_anchor,
                poi=poi_pos, traj_poi=None,
                poi_lengths=batch_x_len, traj_poi_lengths=None
            )
            out_poi_neg = self.model(
                batch_x, batch_x_len, semantic_anchor,
                poi=poi_neg, traj_poi=None,
                poi_lengths=batch_x_len, traj_poi_lengths=None
            )

            # 把 poi_emb 变成 [B,D]（兼容 model 返回 [B,D] 或 [B,T,D]）
            p_poi = self._to_fixed(out_poi_pos.get("poi_emb", None), out_poi_pos["mask"])
            n_poi = self._to_fixed(out_poi_neg.get("poi_emb", None), out_poi_neg["mask"])

            a = anchor_out["traj_repr"]  # [B,D]
            poi_pos_score = self._dot_sim(a, p_poi)
            poi_neg_score = self._dot_sim(a, n_poi)

            l2 = self._bpr_loss(poi_pos_score, poi_neg_score, eps=eps)
            if l2 is not None:
                loss2 = l2
        except Exception as e:
            logging.warning(f"[Trainer] skip loss2 (poi) due to error: {e}")

        # ---- loss3: anchor_traj_repr vs pooled(traj_poi_pos) / pooled(traj_poi_neg) ----
        try:
            out_ctx_pos = self.model(
                batch_x, batch_x_len, semantic_anchor,
                poi=None, traj_poi=batch_traj_poi_pos,
                poi_lengths=None, traj_poi_lengths=batch_x_len
            )
            out_ctx_neg = self.model(
                batch_x, batch_x_len, semantic_anchor,
                poi=None, traj_poi=batch_traj_poi_neg,
                poi_lengths=None, traj_poi_lengths=batch_x_len
            )

            p_ctx = self._to_fixed(out_ctx_pos.get("traj_poi_emb", None), out_ctx_pos["mask"])
            n_ctx = self._to_fixed(out_ctx_neg.get("traj_poi_emb", None), out_ctx_neg["mask"])

            a = anchor_out["traj_repr"]  # [B,D]
            ctx_pos_score = self._dot_sim(a, p_ctx)
            ctx_neg_score = self._dot_sim(a, n_ctx)

            l3 = self._bpr_loss(ctx_pos_score, ctx_neg_score, eps=eps)
            if l3 is not None:
                loss3 = l3
        except Exception as e:
            logging.warning(f"[Trainer] skip loss3 (traj_ctx) due to error: {e}")

        loss = loss1 + loss2 + loss3

        if train:
            torch.backends.cudnn.enabled = False
            (loss / self.grad_accum_steps).backward()

        return float(loss.item())

    def _train_epoch(self):
        self.model.train()
        losses = []
        pbar = tqdm(self.train_data_loader)
        self.optim.zero_grad()
        for step, data in enumerate(pbar, 1):
            loss = self._pass(data, train=True)
            losses.append(loss)
            pbar.set_description("[loss: %f]" % loss)

            if step % self.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.optim.step()
                self.optim.zero_grad()

        # flush remaining gradients
        if len(self.train_data_loader) % self.grad_accum_steps != 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optim.step()
            self.optim.zero_grad()

        return float(np.array(losses).mean())

    def _val_epoch(self):
        self.model.eval()
        if self.val_data_loader is None:
            return None
        losses = []
        pbar = tqdm(self.val_data_loader)
        with torch.no_grad():
            for data in pbar:
                loss = self._pass(data, train=False)
                losses.append(loss)
                pbar.set_description("[val_loss: %f]" % loss)
        return float(np.array(losses).mean()) if losses else None

    def train(self):
        for epoch in range(self.n_epochs):
            train_loss = self._train_epoch()
            logging.info("[Epoch %d/%d] [training loss: %f]" % (epoch, self.n_epochs, train_loss))

            val_loss = self._val_epoch()
            if val_loss is not None:
                logging.info("[Epoch %d/%d] [val loss: %f]" % (epoch, self.n_epochs, val_loss))

            if (epoch + 1) % self.save_epoch_int == 0:
                save_file = os.path.join(self.model_folder, "epoch_%d.pt" % epoch)
                model_to_save = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model
                torch.save(model_to_save.state_dict(), save_file)
                logging.info(f"[Trainer] saved checkpoint -> {save_file}")

## 3. Trainer代码 (`Trainer.py`)
# ```python
# Trainer.py
# import logging
# import os
# import numpy as np
# import torch
# import torch.optim as optim
# from tqdm import tqdm


# class Trainer:
#     def __init__(
#         self,
#         model,
#         train_data_loader,
#         val_data_loader,
#         n_epochs,
#         lr,
#         save_epoch_int,
#         model_folder,
#         device,
#     ):
#         self.train_data_loader = train_data_loader
#         self.val_data_loader = val_data_loader
#         self.n_epochs = n_epochs
#         self.lr = lr
#         self.save_epoch_int = save_epoch_int
#         self.model_folder = model_folder
#         self.device = device
#         self.model = model.to(self.device)

#         if not os.path.exists(model_folder):
#             os.makedirs(model_folder)

#         self.optim = optim.Adam(
#             filter(lambda p: p.requires_grad, self.model.parameters()),
#             lr=lr,
#         )

#         # 🔥 优化2：只在初始化时记录CuDNN状态（不再每step修改）
#         logging.info(f"[Trainer] CuDNN enabled: {torch.backends.cudnn.enabled}")
#         logging.info(f"[Trainer] CuDNN deterministic: {torch.backends.cudnn.deterministic}")

#     # -----------------------------
#     # helpers
#     # -----------------------------
#     @staticmethod
#     def _masked_avg_pool(seq: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
#         """
#         seq: [B, T, D]
#         padding_mask: [B, T], True means padding
#         return: [B, D]
#         """
#         valid = (~padding_mask).float().unsqueeze(-1)  # [B, T, 1]
#         seq = seq * valid
#         denom = valid.sum(dim=1).clamp(min=1.0)        # [B, 1]
#         return seq.sum(dim=1) / denom                  # [B, D]

#     @staticmethod
#     def _to_fixed(x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
#         """
#         ✅ 修复：安全处理mask维度不匹配
#         x could be:
#           - [B, D]  -> return as-is
#           - [B, T, D] -> masked avg pool
#           - None -> None
#         """
#         if x is None:
#             return None
#         if x.dim() == 2:
#             return x
#         if x.dim() == 3:
#             # 如果mask不匹配或为None，创建全False的mask（全部有效）
#             if mask is None or mask.size(1) != x.size(1):
#                 mask = torch.zeros(x.size(0), x.size(1),
#                                   dtype=torch.bool, device=x.device)
#             return Trainer._masked_avg_pool(x, mask)
#         raise ValueError(f"Expected x dim=2 or 3, got {x.shape}")

#     @staticmethod
#     def _dot_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
#         """
#         a,b: [B, D]
#         return: [B]
#         """
#         if a is None or b is None:
#             return None
#         if a.dim() != 2 or b.dim() != 2:
#             raise ValueError(f"Expected [B,D], got a={a.shape}, b={b.shape}")
#         return (a * b).sum(dim=-1)

#     @staticmethod
#     def _bpr_loss(pos_score: torch.Tensor, neg_score: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
#         """
#         BPR: -log(sigmoid(pos-neg))
#         """
#         if pos_score is None or neg_score is None:
#             return None
#         sig = (pos_score - neg_score).sigmoid().clamp(min=eps)
#         return -(sig.log()).mean()

#     def _pass(self, data, train=True):
#         self.optim.zero_grad()

#         (
#             batch_x, batch_n, batch_y,
#             batch_x_len, batch_n_len, batch_y_len,
#             batch_traj_poi_pos, batch_traj_poi_neg,
#             poi_pos, poi_neg,
#             semantic_anchor, semantic_pos, semantic_neg
#         ) = data

#         # -------- move tensors --------
#         batch_x = batch_x.to(self.device)
#         batch_y = batch_y.to(self.device)
#         batch_n = batch_n.to(self.device)

#         poi_pos = poi_pos.to(self.device)
#         poi_neg = poi_neg.to(self.device)
#         batch_traj_poi_pos = batch_traj_poi_pos.to(self.device)
#         batch_traj_poi_neg = batch_traj_poi_neg.to(self.device)

#         # -------- move semantic packs --------
#         semantic_anchor = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in semantic_anchor.items()}
#         semantic_pos    = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in semantic_pos.items()}
#         semantic_neg    = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in semantic_neg.items()}

#         eps = 1e-8

#         # ==========================================================
#         # 1) traj-level: anchor vs pos/neg ✅
#         # ==========================================================
#         anchor_out = self.model(batch_x, batch_x_len, semantic_anchor)
#         pos_out = self.model(batch_y, batch_y_len, semantic_pos)
#         neg_out = self.model(batch_n, batch_n_len, semantic_neg)

#         pos_score_1 = self._dot_sim(anchor_out["traj_repr"], pos_out["traj_repr"])
#         neg_score_1 = self._dot_sim(anchor_out["traj_repr"], neg_out["traj_repr"])
#         loss1 = self._bpr_loss(pos_score_1, neg_score_1, eps=eps)

#         if loss1 is None:
#             loss1 = torch.tensor(0.0, device=self.device)

#         # ==========================================================
#         # 2) poi-level: ✅ 使用新的encode_poi_batch方法
#         # ==========================================================
#         loss2 = torch.tensor(0.0, device=self.device)
#         try:
#             # 直接编码poi_pos/neg（避免重复forward）
#             p_poi = self.model.encode_poi_batch(poi_pos, batch_x_len)
#             n_poi = self.model.encode_poi_batch(poi_neg, batch_x_len)

#             a = anchor_out["traj_repr"]  # [B,D]
#             poi_pos_score = self._dot_sim(a, p_poi)
#             poi_neg_score = self._dot_sim(a, n_poi)

#             l2 = self._bpr_loss(poi_pos_score, poi_neg_score, eps=eps)
#             if l2 is not None:
#                 loss2 = l2
#         except Exception as e:
#             logging.warning(f"[Trainer] skip loss2 (poi) due to error: {e}")

#         # ==========================================================
#         # 3) traj-ctx-level: ✅ 使用新的encode_poi_batch方法
#         # ==========================================================
#         loss3 = torch.tensor(0.0, device=self.device)
#         try:
#             # 直接编码traj_poi_pos/neg
#             p_ctx = self.model.encode_poi_batch(batch_traj_poi_pos, batch_x_len)
#             n_ctx = self.model.encode_poi_batch(batch_traj_poi_neg, batch_x_len)

#             a = anchor_out["traj_repr"]  # [B,D]
#             ctx_pos_score = self._dot_sim(a, p_ctx)
#             ctx_neg_score = self._dot_sim(a, n_ctx)

#             l3 = self._bpr_loss(ctx_pos_score, ctx_neg_score, eps=eps)
#             if l3 is not None:
#                 loss3 = l3
#         except Exception as e:
#             logging.warning(f"[Trainer] skip loss3 (traj_ctx) due to error: {e}")

#         # total loss
#         loss = loss1 + loss2 + loss3

#         if train:
#             # ✅ 优化2：移除cudnn修改，直接训练
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
#             self.optim.step()

#         return float(loss.item())

#     def _train_epoch(self):
#         self.model.train()
#         losses = []
#         pbar = tqdm(self.train_data_loader)
#         for data in pbar:
#             loss = self._pass(data, train=True)
#             losses.append(loss)
#             pbar.set_description("[loss: %f]" % loss)
#         return float(np.array(losses).mean())

#     def _val_epoch(self):
#         self.model.eval()
#         if self.val_data_loader is None:
#             return None
#         losses = []
#         pbar = tqdm(self.val_data_loader)
#         with torch.no_grad():
#             for data in pbar:
#                 loss = self._pass(data, train=False)
#                 losses.append(loss)
#                 pbar.set_description("[val_loss: %f]" % loss)
#         return float(np.array(losses).mean()) if losses else None

#     def train(self):
#         for epoch in range(self.n_epochs):
#             train_loss = self._train_epoch()
#             logging.info("[Epoch %d/%d] [training loss: %f]" % (epoch, self.n_epochs, train_loss))

#             val_loss = self._val_epoch()
#             if val_loss is not None:
#                 logging.info("[Epoch %d/%d] [val loss: %f]" % (epoch, self.n_epochs, val_loss))

#             if (epoch + 1) % self.save_epoch_int == 0:
#                 save_file = os.path.join(self.model_folder, "epoch_%d.pt" % epoch)
#                 torch.save(self.model.state_dict(), save_file)
#                 logging.info(f"[Trainer] saved checkpoint -> {save_file}")

# 版本1---------------------------------------
# # Trainer.py
# import logging
# import os
# import numpy as np
# import torch
# import torch.optim as optim
# from tqdm import tqdm


# class Trainer:
#     def __init__(
#         self,
#         model,
#         train_data_loader,
#         val_data_loader,
#         n_epochs,
#         lr,
#         save_epoch_int,
#         model_folder,
#         device,
#     ):
#         self.train_data_loader = train_data_loader
#         self.val_data_loader = val_data_loader
#         self.n_epochs = n_epochs
#         self.lr = lr
#         self.save_epoch_int = save_epoch_int
#         self.model_folder = model_folder
#         self.device = device
#         self.model = model.to(self.device)

#         if not os.path.exists(model_folder):
#             os.makedirs(model_folder)

#         self.optim = optim.Adam(
#             filter(lambda p: p.requires_grad, self.model.parameters()),
#             lr=lr,
#         )

#     # -----------------------------
#     # helpers: seq -> fixed (兼容 [B,T,D] / [B,D])
#     # -----------------------------
#     @staticmethod
#     def _masked_avg_pool(seq: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
#         """
#         seq: [B, T, D]
#         padding_mask: [B, T], True means padding
#         return: [B, D]
#         """
#         valid = (~padding_mask).float().unsqueeze(-1)  # [B, T, 1]
#         seq = seq * valid
#         denom = valid.sum(dim=1).clamp(min=1.0)        # [B, 1]
#         return seq.sum(dim=1) / denom                  # [B, D]

#     @staticmethod
#     def _to_fixed(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
#         """
#         x could be:
#           - [B, D]  -> return as-is
#           - [B, T, D] -> masked avg pool
#           - None -> None
#         """
#         if x is None:
#             return None
#         if x.dim() == 2:
#             return x
#         if x.dim() == 3:
#             return Trainer._masked_avg_pool(x, mask)
#         raise ValueError(f"Expected x dim=2 or 3, got {x.shape}")

#     @staticmethod
#     def _dot_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
#         """
#         a,b: [B, D]
#         return: [B]
#         """
#         if a is None or b is None:
#             return None
#         if a.dim() != 2 or b.dim() != 2:
#             raise ValueError(f"Expected [B,D], got a={a.shape}, b={b.shape}")
#         return (a * b).sum(dim=-1)

#     @staticmethod
#     def _bpr_loss(pos_score: torch.Tensor, neg_score: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
#         """
#         BPR: -log(sigmoid(pos-neg))
#         """
#         if pos_score is None or neg_score is None:
#             return None
#         sig = (pos_score - neg_score).sigmoid().clamp(min=eps)
#         return -(sig.log()).mean()

#     def _pass(self, data, train=True):
#         self.optim.zero_grad()

#         (
#             batch_x, batch_n, batch_y,
#             batch_x_len, batch_n_len, batch_y_len,
#             batch_traj_poi_pos, batch_traj_poi_neg,
#             poi_pos, poi_neg,
#             semantic_anchor, semantic_pos, semantic_neg
#         ) = data

#         # -------- move tensors --------
#         batch_x = batch_x.to(self.device)
#         batch_y = batch_y.to(self.device)
#         batch_n = batch_n.to(self.device)

#         poi_pos = poi_pos.to(self.device)
#         poi_neg = poi_neg.to(self.device)
#         batch_traj_poi_pos = batch_traj_poi_pos.to(self.device)
#         batch_traj_poi_neg = batch_traj_poi_neg.to(self.device)

#         # -------- move semantic packs --------
#         semantic_anchor = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in semantic_anchor.items()}
#         semantic_pos    = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in semantic_pos.items()}
#         semantic_neg    = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in semantic_neg.items()}

#         eps = 1e-8

#         # ==========================================================
#         # 1) traj-level: anchor vs pos/neg  (固定向量对比) ✅
#         # ==========================================================
#         anchor_out = self.model(batch_x, batch_x_len, semantic_anchor)
#         pos_out = self.model(batch_y, batch_y_len, semantic_pos)
#         neg_out = self.model(batch_n, batch_n_len, semantic_neg)

#         pos_score_1 = self._dot_sim(anchor_out["traj_repr"], pos_out["traj_repr"])
#         neg_score_1 = self._dot_sim(anchor_out["traj_repr"], neg_out["traj_repr"])
#         loss1 = self._bpr_loss(pos_score_1, neg_score_1, eps=eps)

#         if loss1 is None:
#             # 理论上不会发生
#             loss1 = torch.tensor(0.0, device=self.device)

#         # ==========================================================
#         # 2) poi-level / traj-ctx-level:
#         #    关键修复：poi_pos/poi_neg、traj_poi_pos/neg 是按 anchor 轨迹生成的
#         #    所以 loss2/loss3 只围绕 anchor 来做对比（不再用 pos_out/neg_out 的序列）
#         # ==========================================================
#         loss2 = torch.tensor(0.0, device=self.device)
#         loss3 = torch.tensor(0.0, device=self.device)

#         # ---- loss2: anchor_traj_repr vs pooled(poi_pos) / pooled(poi_neg) ----
#         try:
#             # 用同一个 anchor batch_x，但分别喂 poi_pos / poi_neg
#             out_poi_pos = self.model(
#                 batch_x, batch_x_len, semantic_anchor,
#                 poi=poi_pos, traj_poi=None,
#                 poi_lengths=batch_x_len, traj_poi_lengths=None
#             )
#             out_poi_neg = self.model(
#                 batch_x, batch_x_len, semantic_anchor,
#                 poi=poi_neg, traj_poi=None,
#                 poi_lengths=batch_x_len, traj_poi_lengths=None
#             )

#             # 把 poi_emb 变成 [B,D]（兼容 model 返回 [B,D] 或 [B,T,D]）
#             p_poi = self._to_fixed(out_poi_pos.get("poi_emb", None), out_poi_pos["mask"])
#             n_poi = self._to_fixed(out_poi_neg.get("poi_emb", None), out_poi_neg["mask"])

#             a = anchor_out["traj_repr"]  # [B,D]
#             poi_pos_score = self._dot_sim(a, p_poi)
#             poi_neg_score = self._dot_sim(a, n_poi)

#             l2 = self._bpr_loss(poi_pos_score, poi_neg_score, eps=eps)
#             if l2 is not None:
#                 loss2 = l2
#         except Exception as e:
#             logging.warning(f"[Trainer] skip loss2 (poi) due to error: {e}")

#         # ---- loss3: anchor_traj_repr vs pooled(traj_poi_pos) / pooled(traj_poi_neg) ----
#         try:
#             out_ctx_pos = self.model(
#                 batch_x, batch_x_len, semantic_anchor,
#                 poi=None, traj_poi=batch_traj_poi_pos,
#                 poi_lengths=None, traj_poi_lengths=batch_x_len
#             )
#             out_ctx_neg = self.model(
#                 batch_x, batch_x_len, semantic_anchor,
#                 poi=None, traj_poi=batch_traj_poi_neg,
#                 poi_lengths=None, traj_poi_lengths=batch_x_len
#             )

#             p_ctx = self._to_fixed(out_ctx_pos.get("traj_poi_emb", None), out_ctx_pos["mask"])
#             n_ctx = self._to_fixed(out_ctx_neg.get("traj_poi_emb", None), out_ctx_neg["mask"])

#             a = anchor_out["traj_repr"]  # [B,D]
#             ctx_pos_score = self._dot_sim(a, p_ctx)
#             ctx_neg_score = self._dot_sim(a, n_ctx)

#             l3 = self._bpr_loss(ctx_pos_score, ctx_neg_score, eps=eps)
#             if l3 is not None:
#                 loss3 = l3
#         except Exception as e:
#             logging.warning(f"[Trainer] skip loss3 (traj_ctx) due to error: {e}")

#         # total loss
#         loss = loss1 + loss2 + loss3

#         if train:
#             # 你原来关 cudnn 是为了某些可复现/对齐问题；保留也行
#             torch.backends.cudnn.enabled = False
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
#             self.optim.step()

#         return float(loss.item())

#     def _train_epoch(self):
#         self.model.train()
#         losses = []
#         pbar = tqdm(self.train_data_loader)
#         for data in pbar:
#             loss = self._pass(data, train=True)
#             losses.append(loss)
#             pbar.set_description("[loss: %f]" % loss)
#         return float(np.array(losses).mean())

#     def _val_epoch(self):
#         self.model.eval()
#         if self.val_data_loader is None:
#             return None
#         losses = []
#         pbar = tqdm(self.val_data_loader)
#         with torch.no_grad():
#             for data in pbar:
#                 loss = self._pass(data, train=False)
#                 losses.append(loss)
#                 pbar.set_description("[val_loss: %f]" % loss)
#         return float(np.array(losses).mean()) if losses else None

#     def train(self):
#         for epoch in range(self.n_epochs):
#             train_loss = self._train_epoch()
#             logging.info("[Epoch %d/%d] [training loss: %f]" % (epoch, self.n_epochs, train_loss))

#             val_loss = self._val_epoch()
#             if val_loss is not None:
#                 logging.info("[Epoch %d/%d] [val loss: %f]" % (epoch, self.n_epochs, val_loss))

#             if (epoch + 1) % self.save_epoch_int == 0:
#                 save_file = os.path.join(self.model_folder, "epoch_%d.pt" % epoch)
#                 torch.save(self.model.state_dict(), save_file)
#                 logging.info(f"[Trainer] saved checkpoint -> {save_file}")
