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
args.log_path = os.path.join(args.exp_folder, 'train.log')

os.system('mkdir -p ' + args.exp_folder)
os.system('mkdir -p ' + args.ckpt_folder)
os.system('mkdir -p ' + args.train_folder)

tee = Tee(args.log_path, 'w')
print('args\n', args)

args.use_gpu = torch.cuda.is_available()

model = CircuitGNN(args)

if args.resume_epoch > -1:
    model.load_state_dict(torch.load(os.path.join(args.ckpt_folder, 'model_ep%d.pth' % args.resume_epoch)))

model.cuda()
optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
scheduler = MultiStepLR(optimizer, milestones=[500, 700, 900, 1000, 1050, 1100, 1150], gamma=0.5)
print("model #params: %d" % count_parameters(model))

resonator_nums = [4, 5]

datasets = {
    phase: [CircuitDataset(num_block=n, data_root=args.data_root, data_list=load_data_list(args.data_root, f'{phase}_list.txt'))
            for n in resonator_nums]
    for phase in ['train', 'valid']
}

print('Train samples:', [len(dset) for dset in datasets['train']])
print('Valid samples:', [len(dset) for dset in datasets['valid']])

train_loaders = [
    DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    for dataset in datasets['train']
]

valid_loaders = [
    DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    for dataset in datasets['valid']
]


def dump_sample(sample, path):
    pred, label, raw = sample
    fig, ax = plt.subplots(1, 2, figsize=(8, 4), dpi=100)
    plot_circuit(ax[0], raw)
    plot_pred(ax[1], pred)
    plot_label(ax[1], label)
    plt.title('{:.4f} {:.4f}'.format(np.abs(pred[0] - label[0]).mean(), np.abs(pred[1] - label[1]).mean()))
    plt.tight_layout(pad=0)
    plt.savefig(path)
    plt.close()


def train(epoch):
    eps = 1e-3 * 5

    loss_meter = AverageMeter()
    db_meter = AverageMeter()
    l1_r_mtr = AverageMeter()
    l1_i_mtr = AverageMeter()
    l1_m_mtr = AverageMeter()
    mr_m_mtr = AverageMeter()

    model.train()

    optimizer.zero_grad()

    train_iter = mix_iters([iter(loader) for loader in train_loaders])
    train_steps = sum([len(loader) for loader in train_loaders])

    bar = ProgressBar(max_value=train_steps)

    for i, data in bar(enumerate(train_iter)):
        data, label, raw = data
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

        if i % 100 == 0: print('Loss', loss.item(), 'Loss_db', db_loss.item())

        # backward
        db_loss_factor = min(0.2, epoch / 1000)
        total_loss = loss + db_loss * db_loss_factor
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    print('Train: [{0}]\t'
          '#Resonators: [{1}]\t'
          'Loss {loss.avg:.4f}\t Error_db {loss_db.avg:.4f}\n'
          'Error L1 real {l1r.avg:.4f} imag {l1i.avg:.4f} magnitude {l1m.avg:.4f}\n'
          'relative magnitude error {mrm.avg:.4f}'.format(epoch, resonator_nums, loss=loss_meter, loss_db=db_meter, l1r=l1_r_mtr, l1i=l1_i_mtr,
                                                          l1m=l1_m_mtr,
                                                          mrm=mr_m_mtr))

    sys.stdout.flush()

    pred_np = to_np(pred[0])
    label_np = to_np(label[0])
    raw_np = to_np(raw[0])
    dump_sample(sample=(pred_np, label_np, raw_np), path=os.path.join(args.train_folder, 'train_ep{}.png'.format(epoch)))

    return loss_meter.avg, db_meter.avg


def valid(epoch):
    eps = 1e-3 * 5

    model.eval()

    many_loss, many_db = [], []
    for data_loader in valid_loaders:
        loss_meter = AverageMeter()
        db_meter = AverageMeter()
        l1_r_mtr = AverageMeter()
        l1_i_mtr = AverageMeter()
        l1_m_mtr = AverageMeter()
        mr_m_mtr = AverageMeter()

        bar = ProgressBar()

        for i, data in bar(enumerate(data_loader)):
            data, label, raw = data
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

        print('Valid: [{0}]\t'
              '#Resonators: [{1}]\t'
              'Loss {loss.avg:.4f}\t Error_db {loss_db.avg:.4f}\n'
              'Error L1 real {l1r.avg:.4f} imag {l1i.avg:.4f} magnitude {l1m.avg:.4f}\n'
              'relative magnitude error {mrm.avg:.4f}'.format(epoch, data_loader.dataset.n, loss=loss_meter, loss_db=db_meter, l1r=l1_r_mtr, l1i=l1_i_mtr,
                                                              l1m=l1_m_mtr,
                                                              mrm=mr_m_mtr))
        many_loss.append(loss_meter.avg)
        many_db.append(db_meter.avg)

        sys.stdout.flush()

        pred_np = to_np(pred[0])
        label_np = to_np(label[0])
        raw_np = to_np(raw[0])
        dump_sample(sample=(pred_np, label_np, raw_np),
                    path=os.path.join(args.train_folder, 'valid_ep{}_num{}.png'.format(epoch, data_loader.dataset.n)))

    return many_loss, many_db


if __name__ == '__main__':
    best_loss = 10000
    epoch_st = args.resume_epoch + 1

    train_loss_curve, valid_loss_curve = [], []
    train_db_curve, valid_db_curve = [], []

    for epoch in range(epoch_st, epoch_st + 2000):
        train_loss, train_db = train(epoch)
        valid_losses, valid_dbs = valid(epoch)
        scheduler.step()
        valid_loss = np.mean(valid_losses)
        if best_loss > valid_loss:
            best_loss = valid_loss
            print('New best epoch {} valid los {}'.format(epoch,
                                                          valid_loss))
            torch.save(model.state_dict(), os.path.join(args.ckpt_folder, 'model_ep%d.pth' % epoch))

        train_loss_curve.append(train_loss)
        valid_loss_curve.append(valid_losses)

        train_db_curve.append(train_db)
        valid_db_curve.append(valid_dbs)

        f, ax = plt.subplots(1, 2, figsize=(12, 4), dpi=100)
        plot_loss(ax[0], np.array(train_loss_curve), np.array(valid_loss_curve).T, num_list=resonator_nums)
        plot_loss(ax[1], np.array(train_db_curve), np.array(valid_db_curve).T, num_list=resonator_nums)
        plt.savefig(os.path.os.path.join(args.exp_folder, 'loss_curve.png'))
        plt.close()
