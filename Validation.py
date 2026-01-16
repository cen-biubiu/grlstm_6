import numpy as np
import torch
import random
import os
import logging
from tqdm import tqdm

from logg import setup_logger
from pars_args import args
from GRLSTM_Model import GRLSTM
from Data_Loader import TrainDataValLoader, ValValueDataLoader


def build_global_to_trainlocal(train_idx_global: np.ndarray) -> dict:
    # global_id -> local_id (0..len(train_idx)-1)
    return {int(g): int(i) for i, g in enumerate(train_idx_global)}


@torch.no_grad()
def recall_topk_multi_pos(
    s_emb: torch.Tensor,
    train_emb: torch.Tensor,
    pos_local: torch.Tensor,   # [B, P] 每条样本多个正样本（train local index）
    device,
    K=(1, 5, 10, 20, 50),
):
    # scores: [B, Ntrain]
    scores = torch.mm(s_emb, train_emb.t())
    ranked = torch.argsort(scores, dim=1, descending=True)  # [B, Ntrain]

    B = ranked.size(0)
    P = pos_local.size(1)

    # pos_local 里可能有 padding（例如 -1），先过滤掉
    pos_local = pos_local.to(device)
    valid_pos_mask = pos_local >= 0  # [B, P]

    out = torch.zeros((B, len(K)), device=device)

    for i in range(B):
        # 有效正样本集合
        pos_i = pos_local[i][valid_pos_mask[i]]
        if pos_i.numel() == 0:
            continue

        for kk, k in enumerate(K):
            topk_pred = ranked[i, :k]
            # 命中任意一个正样本
            hit = (topk_pred.unsqueeze(1) == pos_i.unsqueeze(0)).any().item()
            if hit:
                out[i, kk:] = 1
                break

    return out


def eval_model_topk():
    device = torch.device(f'cuda:{args.gpu}' if args.gpu >= 0 else 'cpu')
    K = [1, 5, 10, 20, 50]

    # =========================
    # 1) 读取 train / val ground-truth
    # =========================
    train_npz = np.load(args.train_file, allow_pickle=True)
    val_npz = np.load(args.val_file, allow_pickle=True)

    # 训练集必须包含 train_idx（全局索引）
    train_idx_global = train_npz["train_idx"].astype(np.int64)  # [Ntrain]
    # 验证集必须包含 val_pos（全局索引 topP）
    val_pos_global = val_npz["val_pos"].astype(np.int64)        # [Nval, P]

    # 建 global -> train_local 映射
    g2l = build_global_to_trainlocal(train_idx_global)

    # 把 val_pos_global 映射到 train_local（不在 train 的置为 -1）
    Nval, P = val_pos_global.shape
    val_pos_local = np.full((Nval, P), -1, dtype=np.int64)
    for i in range(Nval):
        for p in range(P):
            gid = int(val_pos_global[i, p])
            val_pos_local[i, p] = g2l.get(gid, -1)

    # =========================
    # 2) 准备模型 + dataloader
    # =========================
    model = GRLSTM(args, device, batch_first=True).to(device)

    data_loader_train = TrainDataValLoader(args.train_file, args.batch_size)
    data_loader_val = ValValueDataLoader(args.val_file, args.batch_size)

    emb_train = torch.zeros((len(train_idx_global), args.latent_dim), device=device)
    rec = torch.zeros((Nval, len(K)), device=device)

    # =========================
    # 3) 枚举 checkpoints
    # =========================
    epochs = []
    if os.path.isdir(args.save_folder):
        for f in os.listdir(args.save_folder):
            if f.startswith('epoch_') and f.endswith('.pt'):
                try:
                    epochs.append(int(f[len('epoch_'):-3]))
                except Exception:
                    pass
    epochs.sort()
    if not epochs:
        logging.warning('No checkpoints found in %s' % args.save_folder)
        return

    max_rec_v = -1
    max_epoch = -1

    for epoch in epochs:
        model_f = f"{args.save_folder}/epoch_{epoch}.pt"
        if not os.path.exists(model_f):
            continue

        logging.info('Loading checkpoint from %s' % model_f)
        model.load_state_dict(torch.load(model_f, map_location=device))
        model.eval()

        # --------- 3.1 提取 train embedding（按 train_list local 顺序存）---------
        pbar = tqdm(data_loader_train, desc=f"Embed Train @epoch {epoch}")
        for batch_x, batch_y, batch_x_len, idx_list, semantic in pbar:
            batch_x = batch_x.to(device)
            semantic = {k: v.to(device) if torch.is_tensor(v) else v for k, v in semantic.items()}
            out = model(batch_x, batch_x_len, semantic)
            emb_train[idx_list, :] = out["traj_repr"].detach()

        # --------- 3.2 评测 val Recall@K（multi-pos）---------
        pbar = tqdm(data_loader_val, desc=f"Eval Val @epoch {epoch}")
        for batch_x, batch_y, batch_x_len, idx_list, semantic in pbar:
            batch_x = batch_x.to(device)
            semantic = {k: v.to(device) if torch.is_tensor(v) else v for k, v in semantic.items()}
            out = model(batch_x, batch_x_len, semantic)

            # 取出这个 batch 对应的 val_pos_local
            pos_local_batch = torch.from_numpy(val_pos_local[idx_list]).to(device)

            rec[idx_list, :] = recall_topk_multi_pos(
                out["traj_repr"], emb_train, pos_local_batch, device, K
            ).detach()

        rec_ave = rec.mean(dim=0)
        for v in rec_ave.tolist():
            logging.info('%.4f' % v)

        # 用 Recall@50 做“最佳epoch”也行，你也可以改成 Recall@10
        cur = float(rec_ave[-1].item())
        if cur > max_rec_v:
            max_rec_v = cur
            max_epoch = epoch

    logging.info("Best epoch by Recall@%d = %s" % (K[-1], str(max_epoch)))


if __name__ == '__main__':
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    setup_logger('GRLSTM_eva_topk.log')
    eval_model_topk()
