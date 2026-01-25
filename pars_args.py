import argparse
import os


parser = argparse.ArgumentParser()

# ===================== gpu id ===================== #
parser.add_argument('--gpu',            type=int,   default=0)

# =================== random seed ================== #
parser.add_argument('--seed',           type=int,   default=42)

# ==================== dataset ===================== #
parser.add_argument('--train_file',
                    default='data/data/bj_train_set_topk20.npz')
parser.add_argument('--tra_file',
                    default='data/data/bj_tra.npy')
parser.add_argument('--val_file',
                    default='data/data/bj_val_set_topk20.npz')
parser.add_argument('--test_file',
                    default='data/data/bj_test_set_topk20.npz')
parser.add_argument('--poi_file',
                    default='data/data/bj_transh_poi_10.npz')
parser.add_argument('--semantic_file',
                    default='data/data/bj_node_semantic.txt')
parser.add_argument('--kg_multi_rel_file',
                    default='data/data/bj_KG_graph.txt')
parser.add_argument('--poi_feature_file',
                    default='data/data/poi_features.npy')
parser.add_argument('--topk_neighbors', type=int, default=10)
parser.add_argument('--max_seq_len', type=int, default=1024)
parser.add_argument('--co_heads', type=int, default=4)
parser.add_argument('--gat_hidden_dim', type=int, default=64)
parser.add_argument('--nodes',          type=int,   default=28342,
                    help='Newyork=95581, Beijing=28342')

# ===================== model ====================== #
parser.add_argument('--alpha_spa', type=float, default=0.3,
                    help='spatial weight (align with ground-truth ALPHA)')
parser.add_argument('--delta_sem', type=float, default=0.5,
                    help='semantic weight (align with ground-truth DELTA)')
parser.add_argument('--latent_dim',     type=int,   default=128)
parser.add_argument('--num_heads',      type=int,   default=8)
parser.add_argument('--lstm_layers',    type=int,   default=4)
parser.add_argument('--trans_layers',   type=int,   default=4)
parser.add_argument('--trans_ffn_dim',  type=int,   default=512)
parser.add_argument('--trans_dropout',  type=float, default=0.1)
parser.add_argument('--fusion_layers',  type=int,   default=2)
parser.add_argument('--n_epochs',       type=int,   default=200)
parser.add_argument('--batch_size',     type=int,   default=16)
parser.add_argument('--lr',             type=float, default=5e-4,
                    help='5e-4 for Beijing, 1e-3 for Newyork')
parser.add_argument('--save_epoch_int', type=int,   default=1)
parser.add_argument('--save_folder',                default='saved_models')
parser.add_argument('--grad_accum_steps', type=int, default=1,
                    help='gradient accumulation steps')

# 🔥 轨迹增强对比学习参数 (两视图增强)
parser.add_argument('--use_augmentation', type=bool, default=True,
                    help='whether to use trajectory augmentation')
parser.add_argument('--aug_weight', type=float, default=0.5,
                    help='weight for augmentation contrastive loss (0.3-0.8 recommended)')
parser.add_argument('--aug_temperature', type=float, default=0.05,
                    help='temperature for augmentation InfoNCE loss')

# 增强策略参数
parser.add_argument('--aug_spatial_jitter_ratio', type=float, default=0.3,
                    help='ratio for spatial jitter augmentation')
parser.add_argument('--aug_temporal_interval', type=int, default=2,
                    help='keep interval for temporal resampling (2=keep 50%)')
parser.add_argument('--aug_subsequence_ratio', type=float, default=0.7,
                    help='ratio for subsequence extraction')

args = parser.parse_args()

# setup device
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)