import numpy as np
import torch
import random
import os
import logging

from tqdm import tqdm
from logg import setup_logger
from pars_args import args
from GRLSTM_Model import GRLSTM
from Data_Loader import TestValueDataLoader, TrainDataValLoader


def build_global_to_trainlocal(train_idx_global: np.ndarray) -> dict:
    return {int(g): int(i) for i, g in enumerate(train_idx_global)}


def load_best_epoch_from_log(log_path: str = "GRLSTM_eva.log") -> int:
    # 你原来是取最后3位，这里更稳：找最后一行里能转成 int 的部分
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"Cannot find log file: {log_path}")

    lines = open(log_path, "r").read().splitlines()
    if not lines:
        raise RuntimeError("Empty log file, cannot infer best epoch.")

    last = lines[-1].strip()
    # 尝试直接转
    try:
        return int(last)
    except Exception:
        # 否则抽取最后连续数字
        digits = ""
        for ch in reversed(last):
            if ch.isdigit():
                digits = ch + digits
            elif digits:
                break
        if not digits:
            raise RuntimeError(f"Cannot parse epoch from last log line: {last}")
        return int(digits)


@torch.no_grad()
def recall_multi_pos(
    s_emb: np.ndarray,           # [B, D]
    train_emb: np.ndarray,       # [Ntrain, D]
    pos_local: np.ndarray,       # [B, P]  每条样本多个正样本（train local index），无效为 -1
    K=(1, 5, 10, 20, 50)
) -> np.ndarray:
    r = np.dot(s_emb, train_emb.T)                # [B, Ntrain]
    label_r = np.argsort(-r, axis=1)              # [B, Ntrain]

    out = np.zeros((s_emb.shape[0], len(K)), dtype=np.float32)

    for i in range(s_emb.shape[0]):
        pos_i = pos_local[i]
        pos_i = pos_i[pos_i >= 0]                 # 过滤无效
        if pos_i.size == 0:
            continue
        for kk, k in enumerate(K):
            topk_pred = label_r[i, :k]
            hit = np.intersect1d(topk_pred, pos_i, assume_unique=False).size > 0
            if hit:
                out[i, kk:] = 1.0
                break
    return out


def eval_model():
    device = torch.device(f'cuda:{args.gpu}' if args.gpu >= 0 else 'cpu')

    # ========= 1) 读 train_idx / test_pos（全局索引）=========
    train_npz = np.load(args.train_file, allow_pickle=True)
    test_npz = np.load(args.test_file, allow_pickle=True)

    if "train_idx" not in train_npz:
        raise KeyError(f"{args.train_file} missing key: train_idx (global indices)")
    train_idx_global = train_npz["train_idx"].astype(np.int64)   # [Ntrain]

    # 兼容你可能的命名：test_pos / val_pos
    if "test_pos" in test_npz:
        test_pos_global = test_npz["test_pos"].astype(np.int64)  # [Ntest, P]
    elif "val_pos" in test_npz:
        test_pos_global = test_npz["val_pos"].astype(np.int64)
    else:
        raise KeyError(f"{args.test_file} missing key: test_pos (or val_pos)")

    # 建 global->train_local 映射
    g2l = build_global_to_trainlocal(train_idx_global)

    # 映射 test_pos_global -> test_pos_local（train内部索引）
    Ntest, P = test_pos_global.shape
    test_pos_local = np.full((Ntest, P), -1, dtype=np.int64)
    for i in range(Ntest):
        for p in range(P):
            gid = int(test_pos_global[i, p])
            test_pos_local[i, p] = g2l.get(gid, -1)

    # ========= 2) 准备模型 + 读最佳 epoch =========
    model = GRLSTM(args, device, batch_first=True).to(device)

    best_epoch = load_best_epoch_from_log("GRLSTM_eva.log")
    model_f = f"{args.save_folder}/epoch_{best_epoch}.pt"

    logging.info('Loading best checkpoint from %s' % model_f)
    model.load_state_dict(torch.load(model_f, map_location=device))
    model.eval()

    # ========= 3) dataloader =========
    data_loader_test = TestValueDataLoader(args.test_file, args.batch_size)
    data_loader_train = TrainDataValLoader(args.train_file, args.batch_size)

    # ========= 4) 计算 train embedding（按 train_list local 顺序）=========
    emb_train = np.zeros((len(train_idx_global), args.latent_dim), dtype=np.float32)

    pbar = tqdm(data_loader_train, desc="Embed Train")
    for batch_x, batch_y, batch_x_len, idx_list, semantic in pbar:
        batch_x = batch_x.to(device)
        semantic = {k: v.to(device) if torch.is_tensor(v) else v for k, v in semantic.items()}
        out = model(batch_x, batch_x_len, semantic)
        emb_train[np.asarray(idx_list, dtype=np.int64), :] = out['traj_repr'].detach().cpu().numpy()

    # ========= 5) 测试 Recall@K（命中任意正样本）=========
    K = [1, 5, 10, 20, 50]
    rec = np.zeros((Ntest, len(K)), dtype=np.float32)

    pbar = tqdm(data_loader_test, desc="Eval Test")
    for batch_x, batch_y, batch_x_len, idx_list, semantic in pbar:
        batch_x = batch_x.to(device)
        semantic = {k: v.to(device) if torch.is_tensor(v) else v for k, v in semantic.items()}
        out = model(batch_x, batch_x_len, semantic)

        s_emb = out['traj_repr'].detach().cpu().numpy()
        idx_arr = np.asarray(idx_list, dtype=np.int64)
        pos_local_batch = test_pos_local[idx_arr]  # [B, P]

        rec[idx_arr, :] = recall_multi_pos(s_emb, emb_train, pos_local_batch, K)

    rec_ave = rec.mean(axis=0)
    for v in rec_ave:
        logging.info('%.4f' % v)

    print("Recall@K:", rec_ave)


if __name__ == '__main__':
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    setup_logger('GRLSTM_test_topk.log')
    eval_model()
