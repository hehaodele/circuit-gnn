import sys
import numpy as np


def pickle_save(filename, obj):
    import pickle
    pickle.dump(obj, open(filename, 'wb'), protocol=4)


def pickle_load(filename):
    import pickle
    return pickle.load(open(filename, 'rb'))


class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# ======================================================================================================================
def to_tensor(x):
    import torch
    return torch.tensor(x).to(torch.float).cuda()


def to_np(x):
    return x.detach().cpu().numpy()


def to_db(x):
    import torch
    if isinstance(x, torch.Tensor):
        return 20 * torch.log(x) / np.log(10)
    else:
        return 20 * np.log(x) / np.log(10)


# ======================================================================================================================
class Tee(object):
    def __init__(self, name, mode):
        self.file = open(name, mode)
        self.stdout = sys.stdout
        sys.stdout = self

    def __del__(self):
        sys.stdout = self.stdout
        self.file.close()

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()

    def close(self):
        self.__del__()


# ======================================================================================================================
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def mix_iters(iters):
    table = []
    for i, iter in enumerate(iters):
        table += [i] * len(iter)
    np.random.shuffle(table)
    for i in table:
        yield iters[i].next()


# ======================================================================================================================
"""
plotting for curves
"""


def plot_loss(ax, train, valid_list=None, num_list=None):
    if valid_list is None:
        colors = ['g']
    else:
        from matplotlib import cm
        colors = cm.get_cmap('jet')(np.linspace(0, 1, 1 + len(valid_list)))[:, :3]

    l_train, = ax.plot(train, '-', color=colors[0], alpha=0.8)
    l_train.set_label('train')

    if valid_list is not None:
        for i, (valid, num) in enumerate(zip(valid_list, num_list)):
            l_valid, = ax.plot(valid, '--', color=colors[1 + i], alpha=0.8)
            l_valid.set_label('valid-' + str(num))

    ax.set_xlabel('Epoch')
    ax.set_ylabel('loss')

    ax.legend()
    ax.set_xlim([-1, len(train)])


def plot_label(ax, label):
    ax.plot(np.linspace(200, 400, 5001), label[0], 'r--', alpha=0.8)
    ax.plot(np.linspace(200, 400, 5001), label[1], 'b--', alpha=0.8)
    tmp = np.sqrt(label[1] ** 2 + label[0] ** 2)
    ax.plot(np.linspace(200, 400, 5001), tmp, 'c--', alpha=0.8)
    ax.set_xlim([200, 400])
    ax.set_ylim([-1, 1])


def plot_pred(ax, pred):
    ax.plot(np.linspace(200, 400, 5001), pred[0], 'r-', alpha=0.8)
    ax.plot(np.linspace(200, 400, 5001), pred[1], 'b-', alpha=0.8)
    tmp = np.sqrt(pred[1] ** 2 + pred[0] ** 2)
    ax.plot(np.linspace(200, 400, 5001), tmp, 'c-', alpha=0.8)
    ax.set_xlim([200, 400])
    ax.set_ylim([-1, 1])


def plot_pred_db(ax, pred):
    tmp = np.sqrt(pred[1] ** 2 + pred[0] ** 2)
    tmp = to_db(tmp)
    ax.plot(np.linspace(200, 400, 5001), tmp, 'c-', alpha=0.8)
    ax.set_xlim([200, 400])
    ax.set_ylim([-40, 0])
    ax.set_yticks([-6, -20])


def plot_label_db(ax, pred):
    tmp = np.sqrt(pred[1] ** 2 + pred[0] ** 2)
    tmp = to_db(tmp)
    ax.plot(np.linspace(200, 400, 5001), tmp, 'c--', alpha=0.8)
    ax.set_xlim([200, 400])
    ax.set_ylim([-40, 0])
    ax.set_yticks([-6, -20])


# ======================================================================================================================
"""
plotting for circuits
"""


def plot_resonator(ax, x, y, a, w, xs, ys, dx, dy, i, plot_scale=1):
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Rectangle

    x, y, a, w, xs, ys, dx, dy = [_ * plot_scale for _ in [x, y, a, w, xs, ys, dx, dy]]

    r1 = Rectangle((x - a / 2, y - a / 2), a, a)
    r2 = Rectangle((x - a / 2 + w, y - a / 2 + w), a - w * 2, a - w * 2)
    r3 = Rectangle((xs - dx / 2, ys - dy / 2), dx, dy)

    ax.add_collection(PatchCollection([r1], facecolors='darkorange', alpha=1))
    ax.add_collection(PatchCollection([r2], facecolors='ivory', alpha=1))
    ax.add_collection(PatchCollection([r3], facecolors='ivory', alpha=1))

    ax.text(x, y, str(i), color='k')

    return ax


def is_coupled(x1, y1, x2, y2, a):
    gap_min_ratio = 1.0 / 80
    gap_max_ratio = 1.0 / 5
    dx = abs(x1 - x2)
    dy = abs(y1 - y2)
    sep = max(dx, dy) - a
    shift = min(dx, dy)
    return a * gap_min_ratio <= sep <= a * gap_max_ratio and shift < a


def plot_box(ax, color, x1, y1, x2, y2, lw=2):
    ax.plot([x1, x2], [y1, y1], color + '-', linewidth=lw)
    ax.plot([x1, x2], [y2, y2], color + '-', linewidth=lw)
    ax.plot([x1, x1], [y1, y2], color + '-', linewidth=lw)
    ax.plot([x2, x2], [y1, y2], color + '-', linewidth=lw)


def plot_circuit(ax, para, plot_scale=1, plot_edge=False, opt_box=None):
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Rectangle

    assert para.shape[1] == 9, "para should divide by 9"

    # xmin, xmax = para[:, :2].min() - 10, para[:, :2].max() + 10
    # xmin *= plot_scale
    # xmax *= plot_scale

    para = para.copy()
    xy = para[:, :2]
    xy_mean = xy.mean(0)
    para[:, :2] -= xy_mean[None, :]
    para[:, 4:6] -= xy_mean[None, :]
    a = para[0, 2]
    lim = [xy.min() - a / 2 - 15, xy.max() + a / 2 + 15]
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xticks([])
    ax.set_yticks([])
    # ax.axis('off')

    for i, p in enumerate(para):
        x, y, a, w, xs, ys, dx, dy = p[:8]
        plot_resonator(ax, x, y, a, w, xs, ys, dx, dy, i, plot_scale=plot_scale)

        if opt_box is not None:
            move_max = opt_box[0][i]
            move_dir = opt_box[1][i]
            ax.arrow(x, y, move_dir[0] * 12, move_dir[1] * 12, color='c', width=2)

            x1, y1, x2, y2 = x - a / 2, y - a / 2, x + a / 2, y + a / 2
            x1 = min(x1, x1 + move_dir[0] * move_max)
            y1 = min(y1, y1 + move_dir[1] * move_max)
            x2 = max(x2, x2 + move_dir[0] * move_max)
            y2 = max(y2, y2 + move_dir[1] * move_max)

            plot_box(ax, 'c', x1, y1, x2, y2)

    r_input = Rectangle((para[0, 0] - a / 2 - 20, para[0, 1] - 6), 20, 12)
    r_output = Rectangle((para[-1, 0] + a / 2, para[-1, 1] - 6), 20, 12)

    ax.add_collection(PatchCollection([r_input, r_output], facecolors='darkorange', alpha=1))
    ax.arrow(para[0, 0] - a / 2 - 16, para[0, 1], 12, 0, color='m', width=2)
    ax.arrow(para[-1, 0] + a / 2 - 4, para[-1, 1], 12, 0, color='m', width=2)

    if plot_edge:
        n = len(para)
        for i in range(n):
            for j in range(i + 1, n):
                x1, y1, x2, y2 = para[i, 0], para[i, 1], para[j, 0], para[j, 1],
                if is_coupled(x1, y1, x2, y2, a=para[i, 2]):
                    ax.plot([x1 * plot_scale, x2 * plot_scale], [y1 * plot_scale, y2 * plot_scale], 'k--', lw=2)
