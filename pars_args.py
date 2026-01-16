# import argparse
# import os


# parser = argparse.ArgumentParser()

# # ===================== gpu id ===================== #
# parser.add_argument('--gpu',            type=int,   default=0)

# # =================== random seed ================== #
# parser.add_argument('--seed',           type=int,   default=1234)

# # ==================== dataset ===================== #
# parser.add_argument('--train_file',
#                     default='data/data/bj_train_set.npz')
# parser.add_argument('--val_file',
#                     default='data/data/bj_val_set.npz')
# parser.add_argument('--test_file',
#                     default='data/data/bj_test_set.npz')
# parser.add_argument('--poi_file',
#                     default='data/data/bj_transh_poi_10.npz')
# parser.add_argument('--semantic_file',
#                     default='data/data/bj_node_semantic.txt')
# parser.add_argument('--kg_multi_rel_file',
#                     default='data/data/bj_KG_graph.txt')
# parser.add_argument('--poi_feature_file',
#                     default='data/data/poi_features.npy')
# parser.add_argument('--topk_neighbors', type=int, default=10)
# parser.add_argument('--max_seq_len', type=int, default=512)
# parser.add_argument('--co_heads', type=int, default=4)
# parser.add_argument('--gat_hidden_dim', type=int, default=64)
# parser.add_argument('--nodes',          type=int,   default=28342,
#                     help='Newyork=95581, Beijing=28342')

# # ===================== model ====================== #
# parser.add_argument('--latent_dim',     type=int,   default=128)
# parser.add_argument('--semantic_dim',   type=int,   default=32)
# parser.add_argument('--num_heads',      type=int,   default=8)
# parser.add_argument('--lstm_layers',    type=int,   default=4)
# parser.add_argument('--trans_layers',   type=int,   default=4)
# parser.add_argument('--trans_ffn_dim',  type=int,   default=512)
# parser.add_argument('--trans_dropout',  type=float, default=0.1)
# parser.add_argument('--fusion_layers',  type=int,   default=2)
# parser.add_argument('--n_epochs',       type=int,   default=300)
# parser.add_argument('--batch_size',     type=int,   default=256)
# parser.add_argument('--lr',             type=float, default=5e-4,
#                     help='5e-4 for Beijing, 1e-3 for Newyork')
# parser.add_argument('--save_epoch_int', type=int,   default=1)
# parser.add_argument('--save_folder',                default='saved_models')

# args = parser.parse_args()

# # setup device
# os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

import argparse
import os


parser = argparse.ArgumentParser()

# ===================== gpu id ===================== #
parser.add_argument('--gpu',            type=int,   default=0)

# =================== random seed ================== #
parser.add_argument('--seed',           type=int,   default=42)

# ==================== dataset ===================== #
parser.add_argument('--train_file',
                    default='data/data/bj_train_set_highhit.npz')
parser.add_argument('--tra_file',
                    default='data/data/bj_tra.npy')
parser.add_argument('--val_file',
                    default='data/data/bj_val_set_highhit.npz')
parser.add_argument('--test_file',
                    default='data/data/bj_test_set_highhit.npz')
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
# ===== fixed fusion weights aligned with ground-truth =====
# 使得model中的语义和空间权重和groud-truth中保持一致
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
parser.add_argument('--n_epochs',       type=int,   default=25)
parser.add_argument('--batch_size',     type=int,   default=32)
parser.add_argument('--lr',             type=float, default=5e-4,
                    help='5e-4 for Beijing, 1e-3 for Newyork')
parser.add_argument('--save_epoch_int', type=int,   default=1)
parser.add_argument('--save_folder',                default='saved_models')

args = parser.parse_args()

# setup device
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
