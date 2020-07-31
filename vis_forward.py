import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn.functional as F

import os
import matplotlib.pyplot as plt
from progressbar import ProgressBar

from utils import *

from config import parser, DUMP_FOLDER
from model import CircuitGNN
from dataset import CircuitDataset, load_data_list

args = parser.parse_args()

args.exp_folder = os.path.join(DUMP_FOLDER, args.exp)
args.ckpt_folder = os.path.join(DUMP_FOLDER, args.exp, 'ckpt')
args.train_folder = os.path.join(DUMP_FOLDER, args.exp, 'train')
args.pred_folder = os.path.join(DUMP_FOLDER, args.exp, 'predictions')
os.system('mkdir -p ' + args.pred_folder)
args.log_path = os.path.join(args.exp_folder, f'test_ep{args.epoch}.log')

tee = Tee(args.log_path, 'w')
print('args\n', args)

args.use_gpu = torch.cuda.is_available()

model = CircuitGNN(args)

model.load_state_dict(torch.load(os.path.join(args.ckpt_folder, 'model_ep%d.pth' % args.epoch)))
model.cuda()

max_type = 10


def eval(phase, num_block, circuit_type):
    eps = 1e-3 * 5
    model.eval()

    dataset = CircuitDataset(num_block=num_block, data_root=args.data_root, data_list=load_data_list(args.data_root, f'{phase}_list.txt'), circuit_type=circuit_type, return_fn=True)

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    loss_meter = AverageMeter()
    db_meter = AverageMeter()
    l1_r_mtr = AverageMeter()
    l1_i_mtr = AverageMeter()
    l1_m_mtr = AverageMeter()
    mr_m_mtr = AverageMeter()

    bar = ProgressBar()

    err_list = {
        tp: []
        for tp in range(max_type)
    }

    for i, data in bar(enumerate(data_loader)):
        data, label, raw, fn = data
        node_attr, edge_attr, adj = data
        node_attr, edge_attr, adj, label = [x.cuda() for x in [node_attr, edge_attr, adj, label]]
        pred = model(input=(node_attr, edge_attr, adj))

        loss = F.l1_loss(pred, label)

        # error
        l1_r = torch.abs(pred[:, 0, :] - label[:, 0, :]).mean()
        l1_i = torch.abs(pred[:, 1, :] - label[:, 1, :]).mean()
        mag_pred = torch.sqrt(pred[:, 0, :] ** 2 + pred[:, 1, :] ** 2)
        mag_label = torch.sqrt(label[:, 0, :] ** 2 + label[:, 1, :] ** 2)
        l1_m = torch.abs(mag_label - mag_pred).mean()
        mr_m = (torch.abs(mag_label - mag_pred) / mag_label).mean()

        # db
        db_label = torch.log(torch.clamp(mag_label, eps, 1)) / np.log(10) * 20
        db_pred = torch.log(torch.clamp(mag_pred, eps, 1)) / np.log(10) * 20
        db_loss = F.l1_loss(db_pred, db_label)

        # log
        loss_meter.update(loss.item(), len(label))
        db_meter.update(db_loss.item(), len(label))
        l1_r_mtr.update(l1_r.item(), len(label))
        l1_i_mtr.update(l1_i.item(), len(label))
        l1_m_mtr.update(l1_m.item(), len(label))
        mr_m_mtr.update(mr_m.item(), len(label))

        # fn
        for pred_np, fname, db_p, db_l, r in zip(to_np(pred), fn, to_np(db_pred), to_np(db_label), to_np(raw)):
            from matplotlib import gridspec

            fn_folder, fn_pkl = fname.split('/')[-2:]

            fig = plt.figure(figsize=(10, 3), dpi=200)
            gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.7])
            ax = [plt.subplot(gs[0]), plt.subplot(gs[1])]
            plot_circuit(ax[0], para=r)
            ax[0].set_aspect(1)
            l1, = ax[1].plot(np.linspace(200, 400, 5001), db_l, 'g--', alpha=0.5, lw=3)
            l2, = ax[1].plot(np.linspace(200, 400, 5001), db_p, 'g-', alpha=0.5, lw=3)
            ax[1].legend([l1, l2], [r'$|s_{21}|$ Ground Truth', r'$|s_{21}|$ Circuit-GNN'],
                         loc='best')
            ax[1].set_ylim([-60, 0])
            ax[1].set_yticks([-40, -30, -20, -10])
            ax[1].set_ylabel(r'$|s_{21}|$ (db)')
            ax[1].set_xlabel('Frequency (GHz)')
            ax[1].set_title(phase + ": " + fn_folder + '/' + fn_pkl)
            plt.tight_layout(pad=0)
            plt.show()

    return err_list


if __name__ == '__main__':
    eval(args.phase, num_block=args.num_resonator, circuit_type=args.circuit_type)
