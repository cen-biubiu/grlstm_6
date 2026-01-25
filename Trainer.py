# 改进的图对比学习
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
            aug_weight: float = 0.5,
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
        self.aug_weight = aug_weight

        if not os.path.exists(model_folder):
            os.makedirs(model_folder)

        self.optim = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr,
        )

        logging.info(f"[Trainer] Augmentation contrastive weight: {aug_weight}")

    @staticmethod
    def _masked_avg_pool(seq: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        valid = (~padding_mask).float().unsqueeze(-1)
        seq = seq * valid
        denom = valid.sum(dim=1).clamp(min=1.0)
        return seq.sum(dim=1) / denom

    @staticmethod
    def _to_fixed(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if x is None:
            return None
        if x.dim() == 2:
            return x
        if x.dim() == 3:
            return Trainer._masked_avg_pool(x, mask)
        raise ValueError(f"Expected x dim=2 or 3, got {x.shape}")

    @staticmethod
    def _dot_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if a is None or b is None:
            return None
        if a.dim() != 2 or b.dim() != 2:
            raise ValueError(f"Expected [B,D], got a={a.shape}, b={b.shape}")
        return (a * b).sum(dim=-1)

    @staticmethod
    def _bpr_loss(pos_score: torch.Tensor, neg_score: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        if pos_score is None or neg_score is None:
            return None
        sig = (pos_score - neg_score).sigmoid().clamp(min=eps)
        return -(sig.log()).mean()

    @staticmethod
    def _triplet_loss_with_margin(
            anchor: torch.Tensor,
            pos: torch.Tensor,
            neg: torch.Tensor,
            margin: float = 0.5,  # 🔥 关键: 增大margin
            temperature: float = 0.1  # 🔥 降低温度增强判别力
    ) -> torch.Tensor:
        """Triplet loss with adaptive margin"""
        import torch.nn.functional as F

        # L2 normalize
        anchor = F.normalize(anchor, p=2, dim=-1)
        pos = F.normalize(pos, p=2, dim=-1)
        neg = F.normalize(neg, p=2, dim=-1)

        # Cosine similarity
        pos_sim = (anchor * pos).sum(dim=-1) / temperature
        neg_sim = (anchor * neg).sum(dim=-1) / temperature

        # Triplet loss: max(0, margin + neg_sim - pos_sim)
        loss = F.relu(margin + neg_sim - pos_sim).mean()

        return loss

    @staticmethod
    def _listwise_ranking_loss(
            anchor: torch.Tensor,  # [B, D]
            pos: torch.Tensor,  # [B, D]
            neg: torch.Tensor,  # [B, D]
            temperature: float = 0.05
    ) -> torch.Tensor:
        """
        Listwise ranking loss - 强制正样本排在第一位
        """
        import torch.nn.functional as F

        B = anchor.size(0)

        # Normalize
        anchor = F.normalize(anchor, p=2, dim=-1)
        pos = F.normalize(pos, p=2, dim=-1)
        neg = F.normalize(neg, p=2, dim=-1)

        # Scores
        pos_scores = (anchor * pos).sum(dim=-1, keepdim=True) / temperature  # [B, 1]
        neg_scores = (anchor * neg).sum(dim=-1, keepdim=True) / temperature  # [B, 1]

        # Concatenate [pos, neg]
        scores = torch.cat([pos_scores, neg_scores], dim=1)  # [B, 2]

        # Labels: 正样本应该排第一 (index=0)
        labels = torch.zeros(B, dtype=torch.long, device=anchor.device)

        # Cross entropy - 强制pos_score > neg_score
        loss = F.cross_entropy(scores, labels)

        return loss

    @staticmethod
    def _infonce_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.05) -> torch.Tensor:
        """InfoNCE对比损失 (两视图对比学习)"""
        import torch.nn.functional as F

        B = z1.size(0)
        z1 = F.normalize(z1, p=2, dim=-1)
        z2 = F.normalize(z2, p=2, dim=-1)

        sim = torch.mm(z1, z2.t()) / temperature
        labels = torch.arange(B, device=z1.device)

        loss_12 = F.cross_entropy(sim, labels)
        loss_21 = F.cross_entropy(sim.t(), labels)

        return (loss_12 + loss_21) / 2

    def _pass(self, data, train=True):
        # 解包 (包含两视图增强数据)
        if len(data) == 20:  # 有增强
            (
                batch_x, batch_n, batch_y,
                batch_x_len, batch_n_len, batch_y_len,
                batch_traj_poi_pos, batch_traj_poi_neg,
                poi_pos, poi_neg,
                semantic_anchor, semantic_pos, semantic_neg,
                batch_x_view1, batch_x_view2,
                view1_len, view2_len,
                semantic_view1, semantic_view2,
                aug_strategies,
            ) = data
            has_aug = True
        else:  # 无增强 (向后兼容)
            (
                batch_x, batch_n, batch_y,
                batch_x_len, batch_n_len, batch_y_len,
                batch_traj_poi_pos, batch_traj_poi_neg,
                poi_pos, poi_neg,
                semantic_anchor, semantic_pos, semantic_neg
            ) = data
            has_aug = False

        batch_x = batch_x.to(self.device)
        batch_y = batch_y.to(self.device)
        batch_n = batch_n.to(self.device)

        poi_pos = poi_pos.to(self.device)
        poi_neg = poi_neg.to(self.device)
        batch_traj_poi_pos = batch_traj_poi_pos.to(self.device)
        batch_traj_poi_neg = batch_traj_poi_neg.to(self.device)

        semantic_anchor = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in semantic_anchor.items()}
        semantic_pos = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in semantic_pos.items()}
        semantic_neg = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in semantic_neg.items()}

        eps = 1e-8

        # ==========================================================
        # 1) traj-level对比
        # ==========================================================
        anchor_out = self.model(batch_x, batch_x_len, semantic_anchor)
        pos_out = self.model(batch_y, batch_y_len, semantic_pos)
        neg_out = self.model(batch_n, batch_n_len, semantic_neg)

        pos_score_1 = self._dot_sim(anchor_out["traj_repr"], pos_out["traj_repr"])
        neg_score_1 = self._dot_sim(anchor_out["traj_repr"], neg_out["traj_repr"])
        # loss1 = self._bpr_loss(pos_score_1, neg_score_1, eps=eps)
        loss1 = self._triplet_loss_with_margin(
            anchor_out["traj_repr"],
            pos_out["traj_repr"],
            neg_out["traj_repr"],
            margin=0.5,  # 可调参数: 0.3-0.8
            temperature=0.1
        )
        loss_ranking = self._listwise_ranking_loss(
            anchor_out["traj_repr"],
            pos_out["traj_repr"],
            neg_out["traj_repr"],
            temperature=0.05  # 更低的温度 = 更严格的排序
        )

        if loss1 is None:
            loss1 = torch.tensor(0.0, device=self.device)

        # ==========================================================
        # 2) POI-level对比
        # ==========================================================
        loss2 = torch.tensor(0.0, device=self.device)
        try:
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

            p_poi = self._to_fixed(out_poi_pos.get("poi_emb", None), out_poi_pos["mask"])
            n_poi = self._to_fixed(out_poi_neg.get("poi_emb", None), out_poi_neg["mask"])

            a = anchor_out["traj_repr"]
            poi_pos_score = self._dot_sim(a, p_poi)
            poi_neg_score = self._dot_sim(a, n_poi)

            l2 = self._bpr_loss(poi_pos_score, poi_neg_score, eps=eps)
            if l2 is not None:
                loss2 = l2
        except Exception as e:
            logging.warning(f"[Trainer] skip loss2 (poi) due to error: {e}")

        # ==========================================================
        # 3) traj-ctx-level对比
        # ==========================================================
        loss3 = torch.tensor(0.0, device=self.device)
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

            a = anchor_out["traj_repr"]
            ctx_pos_score = self._dot_sim(a, p_ctx)
            ctx_neg_score = self._dot_sim(a, n_ctx)

            l3 = self._bpr_loss(ctx_pos_score, ctx_neg_score, eps=eps)
            if l3 is not None:
                loss3 = l3
        except Exception as e:
            logging.warning(f"[Trainer] skip loss3 (traj_ctx) due to error: {e}")

        # ==========================================================
        #  4) 两视图增强对比学习
        # ==========================================================
        loss_aug = torch.tensor(0.0, device=self.device)
        loss_aug_dict = {}

        if has_aug:
            try:
                batch_x_view1 = batch_x_view1.to(self.device)
                batch_x_view2 = batch_x_view2.to(self.device)
                semantic_view1 = {k: v.to(self.device) if torch.is_tensor(v) else v
                                  for k, v in semantic_view1.items()}
                semantic_view2 = {k: v.to(self.device) if torch.is_tensor(v) else v
                                  for k, v in semantic_view2.items()}

                # 编码两个增强视图
                view1_out = self.model(batch_x_view1, view1_len, semantic_view1)
                view2_out = self.model(batch_x_view2, view2_len, semantic_view2)

                # 对比损失: anchor vs view1, anchor vs view2, view1 vs view2
                loss_anchor_view1 = self._infonce_loss(
                    anchor_out["traj_repr"],
                    view1_out["traj_repr"],
                    temperature=0.07
                )

                loss_anchor_view2 = self._infonce_loss(
                    anchor_out["traj_repr"],
                    view2_out["traj_repr"],
                    temperature=0.07
                )

                loss_view1_view2 = self._infonce_loss(
                    view1_out["traj_repr"],
                    view2_out["traj_repr"],
                    temperature=0.07
                )

                # 总增强损失
                loss_aug = (loss_anchor_view1 + loss_anchor_view2 + loss_view1_view2) / 3

                loss_aug_dict = {
                    'aug_anchor_v1': float(loss_anchor_view1.item()),
                    'aug_anchor_v2': float(loss_anchor_view2.item()),
                    'aug_v1_v2': float(loss_view1_view2.item()),
                }

                # 统计增强策略使用情况
                strategy_counts = {}
                for s in aug_strategies:
                    strategy_counts[s] = strategy_counts.get(s, 0) + 1
                loss_aug_dict['strategies'] = strategy_counts

            except Exception as e:
                logging.warning(f"[Trainer] skip augmentation loss due to error: {e}")

        # 总损失
        loss = loss1 + loss2 + loss3 + self.aug_weight * loss_aug + 0.3 * loss_ranking

        if train:
            torch.backends.cudnn.enabled = False
            (loss / self.grad_accum_steps).backward()

        # 返回损失字典
        loss_dict = {
            'total': float(loss.item()),
            'traj': float(loss1.item()),
            'poi': float(loss2.item()),
            'ctx': float(loss3.item()),
            'aug': float(loss_aug.item()),
        }
        loss_dict.update(loss_aug_dict)

        return loss_dict

    def _train_epoch(self):
        self.model.train()
        losses = []
        aug_losses = []

        pbar = tqdm(self.train_data_loader)
        self.optim.zero_grad()

        for step, data in enumerate(pbar, 1):
            loss_dict = self._pass(data, train=True)
            losses.append(loss_dict['total'])
            aug_losses.append(loss_dict['aug'])

            # 显示使用的增强策略
            strategy_info = ""
            if 'strategies' in loss_dict:
                strategy_info = " | " + ", ".join(
                    f"{k}:{v}" for k, v in loss_dict['strategies'].items()
                )

            pbar.set_description(
                "[loss: %.4f | aug: %.4f%s]" %
                (loss_dict['total'], loss_dict['aug'], strategy_info)
            )

            if step % self.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.optim.step()
                self.optim.zero_grad()

        if len(self.train_data_loader) % self.grad_accum_steps != 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optim.step()
            self.optim.zero_grad()

        avg_loss = float(np.array(losses).mean())
        avg_aug = float(np.array(aug_losses).mean())

        return avg_loss, avg_aug

    def _val_epoch(self):
        self.model.eval()
        if self.val_data_loader is None:
            return None, None

        losses = []
        aug_losses = []
        pbar = tqdm(self.val_data_loader)

        with torch.no_grad():
            for data in pbar:
                loss_dict = self._pass(data, train=False)
                losses.append(loss_dict['total'])
                aug_losses.append(loss_dict['aug'])
                pbar.set_description("[val_loss: %.4f]" % loss_dict['total'])

        avg_loss = float(np.array(losses).mean()) if losses else None
        avg_aug = float(np.array(aug_losses).mean()) if aug_losses else None

        return avg_loss, avg_aug

    def train(self):
        for epoch in range(self.n_epochs):
            train_loss, train_aug = self._train_epoch()
            logging.info(
                "[Epoch %d/%d] [train loss: %.4f | aug: %.4f]" %
                (epoch, self.n_epochs, train_loss, train_aug)
            )

            val_loss, val_aug = self._val_epoch()
            if val_loss is not None:
                logging.info(
                    "[Epoch %d/%d] [val loss: %.4f | aug: %.4f]" %
                    (epoch, self.n_epochs, val_loss, val_aug)
                )

            if (epoch + 1) % self.save_epoch_int == 0:
                save_file = os.path.join(self.model_folder, "epoch_%d.pt" % epoch)
                model_to_save = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model
                torch.save(model_to_save.state_dict(), save_file)
                logging.info(f"[Trainer] saved checkpoint -> {save_file}")