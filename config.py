from easydict import EasyDict
import argparse

# ======================================================================================================================
CONST = EasyDict()

CONST.a_range = [50, 100]
CONST.w = 6
CONST.open_len = 2.4
CONST.gap_min_ratio = 1.0 / 80
CONST.gap_max_ratio = 1.0 / 5

CONST.raw_data_names = ['X', 'Y', 'a', 'w', 'xs', 'ys', 'dx', 'dy', 'u']

# ======================================================================================================================
DUMP_FOLDER = './dump'
DEFAULTS = EasyDict()
DEFAULTS.num_workers = 20

parser = argparse.ArgumentParser()
# exp
parser.add_argument('--exp', type=str, default='exp_example', help="Exp id")
parser.add_argument('--data_root', type=str, default='./data', help="Path to the dataset folder")
parser.add_argument('--use_gpu', type=bool, default=True, )

# model
parser.add_argument('--len_hidden', type=int, default=400, help='Number of hidden units.')
parser.add_argument('--len_hidden_predictor', type=int, default=512, help='Number of predictor hidden units.')
parser.add_argument('--len_node_attr', type=int, default=11, help='Length of node attributes.')
parser.add_argument('--len_edge_attr', type=int, default=20, help='Length of edge attributes.')
parser.add_argument('--gnn_layers', type=int, default=4, help='Number of GNN layers')
parser.add_argument('--dropout', type=float, default=0, help='dropout model units')

# train
parser.add_argument('--num_workers', type=int, default=DEFAULTS.num_workers, help="")
parser.add_argument('--batch_size', type=int, default=256, help="")
parser.add_argument('--lr', type=float, default=2e-4, help="")
parser.add_argument('--resume_epoch', type=int, default=-1, help="Resume training")

# test
parser.add_argument('--epoch', type=int, default=501, help="Model checkpoint epoch")

# vis_forward
parser.add_argument('--num_resonator', type=int, default=4, help="Number of resonators in the circuit")
parser.add_argument('--circuit_type', type=int, default=2, help="Topology type of the circuit")
parser.add_argument('--phase', type=str, default='test', help="[test, valid, train]")

# ======================================================================================================================
