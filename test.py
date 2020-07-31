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


def eval(phase, num_block):
    eps = 1e-3 * 5
    model.eval()

    dataset = CircuitDataset(num_block=num_block, data_root=args.data_root, data_list=load_data_list(args.data_root, f'{phase}_list.txt'), return_fn=True)

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
        for pred_np, fname, db_p, db_l in zip(to_np(pred), fn, to_np(db_pred), to_np(db_label)):
            fn_folder, fn_pkl = fname.split('/')[-2:]
            # os.system('mkdir -p ' + args.pred_folder + '/' + fn_folder)
            # pickle_save(args.pred_folder + '/' + fn_folder + '/' + fn_pkl, pred_np)
            tp = int(fn_folder.split('_')[-1][4:])
            err_list[tp] += [np.abs(db_p - db_l).mean()]

    print('[{0}]\t'
          '#Resonators: [{1}]\t'
          'Loss {loss.avg:.4f}\t Error_db {loss_db.avg:.4f}\n'
          'Error L1 real {l1r.avg:.4f} imag {l1i.avg:.4f} magnitude {l1m.avg:.4f}\n'
          'relative magnitude error {mrm.avg:.4f}'.format(phase, data_loader.dataset.n, loss=loss_meter, loss_db=db_meter, l1r=l1_r_mtr, l1i=l1_i_mtr,
                                                          l1m=l1_m_mtr,
                                                          mrm=mr_m_mtr))

    return err_list


if __name__ == '__main__':
    from prettytable import PrettyTable

    def gen_mean_err(err):
        err = np.array(err)
        err = err[np.isnan(err) == 0]
        if len(err) > 0:
            return f'{np.mean(err):.03f}'
        else:
            return '-'

    table = PrettyTable(['# of resonator', 'topology', '# of samples', 'train error (db)', 'valid error (db)', 'test error (db)'])

    for num_resonator in [4, 5, 3, 6]:

        test_err = eval('test', num_resonator)
        valid_err = eval('valid', num_resonator)
        train_err = eval('train', num_resonator)

        avg_err = {
            'test': [],
            'valid': [],
            'train': [],
        }
        for circuit_type in range(max_type):
            if len(test_err[circuit_type]) == 0:
                continue
            row = [num_resonator, circuit_type, f'{len(train_err[circuit_type])} {len(valid_err[circuit_type])} {len(test_err[circuit_type])}']
            if len(train_err[circuit_type]) > 0:
                row += [f'{np.mean(train_err[circuit_type]):.03f}']
            else:
                row += '-'

            if len(valid_err[circuit_type]) > 0:
                row += [f'{np.mean(valid_err[circuit_type]):.03f}']
            else:
                row += '-'

            row += [f'{np.mean(test_err[circuit_type]):.03f}']

            table.add_row(row)

            avg_err['train'] += [np.mean(train_err[circuit_type])]
            avg_err['valid'] += [np.mean(valid_err[circuit_type])]
            avg_err['test'] += [np.mean(test_err[circuit_type])]

        row = [num_resonator, 'avg', '-', gen_mean_err(avg_err['train']), gen_mean_err(avg_err['valid']), gen_mean_err(avg_err['test'])]
        table.add_row(row)

    print(table)
