from utils import *
from model import CircuitGNN
from config import parser, DUMP_FOLDER, CONST
from dataset import STATS
from circuit import CircuitGenerator
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import time
import matplotlib.pyplot as plt


# ======================================================================================================================
def to_ghz(x):
    return x / 5001.0 * 200 + 200


def plot_band(ax, band):
    l, r = band
    ax.axvspan(l, r, alpha=0.5, color='gold', label='Target Pass-band')


def plot_pred_band(ax, band):
    l, r = band
    ax.axvspan(l, r, alpha=0.3, color='lime', label='Model predicted Pass-band')


def get_pass_band(y):
    s = np.sqrt(y[0] ** 2 + y[1] ** 2)
    s = to_db(s)
    idx = np.argmax(s)
    s_max = np.max(s)
    l, r = idx, idx
    while l > 0 and s[l - 1] > s_max - 3: l -= 1
    while r + 1 < len(s) and s[r + 1] > s_max - 3: r += 1
    return (to_ghz(l), to_ghz(r)), s_max


def get_iou(x, y):
    return max(0, (min(x[1], y[1]) - max(x[0], y[0])) / (max(x[1], y[1]) - min(x[0], y[0])))


def get_u_info(u):
    u_side = np.zeros((len(u), 4), dtype=np.float32)
    u_shift = np.zeros((len(u), 4), dtype=np.float32)

    idx = (0 <= u) & (u < 1.0 / 8)  # right open
    u_side[idx, 0] = 1
    u_shift[idx, 0] = u[idx]

    idx = (7.0 / 8 <= u) & (u <= 1)  # right open still
    u_side[idx, 0] = 1
    u_shift[idx, 0] = u[idx] - 1

    idx = (1.0 / 8 <= u) & (u < 3.0 / 8)  # up open
    u_side[idx, 1] = 1
    u_shift[idx, 1] = u[idx] - 2.0 / 8

    idx = (3.0 / 8 <= u) & (u < 5.0 / 8)  # left open
    u_side[idx, 2] = 1
    u_shift[idx, 2] = u[idx] - 4.0 / 8

    idx = (5.0 / 8 <= u) & (u < 7.0 / 8)  # down open
    u_side[idx, 3] = 1
    u_shift[idx, 3] = u[idx] - 6.0 / 8

    assert (u_side.sum(1) == 1).all()
    assert (u_shift <= 1.0 / 8).all()
    assert (u_shift >= - 1.0 / 8).all()

    u_info = np.concatenate([u_side, u_shift], 1)

    # print('u', u)
    # print('u_info', u_info)

    return u_info


# ======================================================================================================================

args = parser.parse_args()

args.exp_folder = os.path.join(DUMP_FOLDER, args.exp)
args.ckpt_folder = os.path.join(DUMP_FOLDER, args.exp, 'ckpt')
args.log_path = os.path.join(args.exp_folder, 'inverse.log')

model = CircuitGNN(args)
model_path = os.path.join(args.ckpt_folder, 'model_ep%d.pth' % args.epoch)
print('loading model from ', model_path)
model.load_state_dict(torch.load(os.path.join(args.ckpt_folder, 'model_ep%d.pth' % args.epoch)))
model.cuda()
model.eval()


# ======================================================================================================================

def construct_objective(band, cutoff_width=0.5,
                        alpha=0.5, beta=0.75):
    """
    :param band: the range of passband from band[0] GHz to band[1] GHz
    :param cutoff_width: the transition region from PASS to SUPRESS
    :param alpha: loss weighting for pass band and non-pass band
    :param beta: loss weighting for near non-pass band and far non-pass band
    :return:
    """
    l, r = band
    bw = r - l

    l, r, ll, rr = l + bw * 0.15, r - bw * 0.15, l - bw * 0.01, r + bw * 0.01
    lll, rrr = ll - bw * cutoff_width, rr + bw * cutoff_width

    l, r, ll, rr, lll, rrr = [int((x - 200) / (400 - 200) * 5001) for x in [l, r, ll, rr, lll, rrr]]

    def obj(x):
        # in the passband we want 'x' to be 1
        loss_pass = ((1 - x[:, l:r]) ** 2).mean(1)
        # ous the passband we want 'x' to be 0
        loss_cutoff = ((x[:, lll:ll] ** 2).sum(1) + (x[:, rr:rrr] ** 2).sum(1)) / (ll - lll + rrr - rr)
        loss_faroff = ((x[:, :lll] ** 2).sum(1) + (x[:, rrr:] ** 2).sum(1)) / (lll - 0 + 5001 - rrr)
        # use beta the balance the error of the region near the passband and far from the passband
        # beta somehow control the sharpness of the cutoff
        loss_off = loss_cutoff * beta + loss_faroff * (1 - beta)
        loss = loss_pass * alpha + loss_off * (1 - alpha)
        return loss

    return obj


# ======================================================================================================================
def prepare_input_batch(para):
    """
    :param xy, a, u_info: B x N x len_feature
    :return:
    """

    xy, a, u_info = para
    a = a[:, :, 0]
    B, N = a.size(0), a.size(1)

    '''
    node_attr: B x N x 11
    '''
    is_input = np.zeros((B, N))
    is_output = np.zeros((B, N))
    is_input[:, 0] = 1
    is_output[:, -1] = 1
    is_input, is_output = to_tensor(is_input), to_tensor(is_output)
    node_attr = torch.cat([is_input[:, :, None], is_output[:, :, None], a[:, :, None], u_info], 2)

    '''
    edge_attr: B x N x N x 20
    '''
    xy_diff = xy[:, :, None, :] - xy[:, None, :, :]
    xy_diff_abs = torch.abs(xy_diff)

    u1_info = u_info[:, :, None, :].repeat(1, 1, N, 1)
    u2_info = u_info[:, None, :, :].repeat(1, N, 1, 1)

    gap = torch.max(xy_diff_abs[:, :, :, 0], xy_diff_abs[:, :, :, 1]) - a[:, :, None]
    shift = torch.min(xy_diff_abs[:, :, :, 0], xy_diff_abs[:, :, :, 1])

    gap_a = gap / a[:, :, None]
    gap_a.transpose(0, 2).diagonal()[:] = 1

    gap_a = torch.clamp(gap_a, min=CONST.gap_min_ratio)
    shift_a = shift / a[:, :, None]
    log_gap_a = torch.log(gap_a)

    edge_attr = torch.cat([u1_info, u2_info, xy_diff, log_gap_a[:, :, :, None], shift_a[:, :, :, None]], 3)

    '''
    adj
    '''
    alen = a.detach()
    nearlen = (alen * 1.2).detach()
    xy_dist = torch.sqrt(xy_diff[:, :, :, 0] ** 2 + xy_diff[:, :, :, 1] ** 2 + 1e-3)
    distlen = xy_dist - nearlen[:, :, None]
    distlen = F.relu(distlen)
    distlen = distlen * 2 + nearlen[:, :, None]
    adj = nearlen[:, :, None] / distlen
    adj = torch.clamp(adj, 0.0, 1.0)
    adj.transpose(0, 2).diagonal()[:] = 0

    '''normalize'''
    node_attr = normalize(node_attr, STATS_VAR.node_attr)
    edge_attr = normalize(edge_attr, STATS_VAR.edge_attr)

    return node_attr, edge_attr, adj


# ======================================================================================================================
from easydict import EasyDict

STATS_VAR = EasyDict()
STATS_VAR.node_attr = to_tensor(STATS.node_attr)
STATS_VAR.edge_attr = to_tensor(STATS.edge_attr)


def normalize(x, stat, inverse=False):
    tmp_shape = x.shape
    tmp_len = x.shape[-1]
    if not inverse:
        return ((x.reshape(-1, tmp_len) - stat[:, 0][None, :]) / stat[:, 1][None, :]).reshape(tmp_shape)
    else:
        return (x.reshape(-1, tmp_len) * stat[:, 1][None, :] + stat[:, 0][None, :]).reshape(tmp_shape)


# ======================================================================================================================
def point_in_box(x1, x2, y1, y2, u, v):
    return (x1 < u) & (u < x2) & (y1 < v) & (v < y2)


def box_collision_batch(x1, x2, y1, y2, u1, u2, v1, v2):
    return point_in_box(x1, x2, y1, y2, u1, v1) | point_in_box(x1, x2, y1, y2, u1, v2) | \
           point_in_box(x1, x2, y1, y2, u2, v1) | point_in_box(x1, x2, y1, y2, u2, v2)


def atanh(x):
    return 0.5 * np.log((1 + x) / (1 - x))


def get_u(u_info):
    u_side = u_info[:, :, :4]
    u_shift = u_info[:, :, 4:]

    u = np.zeros((u_side.shape[0], u_side.shape[1]))

    idx = u_side[:, :, 0] > 0
    u[idx] = u_shift[:, :, 0][idx]

    idx = u_side[:, :, 1] > 0
    u[idx] = u_shift[:, :, 1][idx] + 1.0 / 4

    idx = u_side[:, :, 2] > 0
    u[idx] = u_shift[:, :, 2][idx] + 2.0 / 4

    idx = u_side[:, :, 3] > 0
    u[idx] = u_shift[:, :, 3][idx] + 3.0 / 4

    u[u < 0] = u[u < 0] + 1
    return u


class InverseDesigner(object):

    def __init__(self):
        self.obj = None
        self.model = None

        self.raw = None
        self.raw_xy = None
        self.raw_fix = None
        self.raw_a = None
        self.w = None
        self.u_info = None

        self.xy_parameter = None
        self.u_parameter = None
        self.parameter_factor = None
        self.parameter_directions = None
        self.optimizer = None
        self.lr = 3e-2

        self.target_band = None
        self.label_goal = None

    def set_objective(self, obj):
        self.obj = obj

    def set_model(self, model):
        self.model = model

    def parameterize(self, raw):
        """
        :param raw: np.array
            ['X', 'Y', 'a', 'w', 'xs', 'ys', 'dx', 'dy', 'u']
            B x N x 9
        :return:
        """

        B, N = raw.shape[0], raw.shape[1]
        x, y, a, w, u = raw[:, :, 0], raw[:, :, 1], raw[:, :, 2], raw[:, :, 3], raw[:, :, -1]
        self.w = w

        u_info = get_u_info(u.reshape(B * N)).reshape(B, N, 8)

        max_gap = a[:, 0] * CONST.gap_max_ratio
        min_gap = a[:, 0] * CONST.gap_min_ratio
        # print('max min gap', max_gap.shape, min_gap.shape)

        four_directions = np.array(
            [[1, 0], [0, -1], [-1, 0], [0, 1]]
        )

        direction_indexes = np.random.randint(4, size=(B, N))
        directions = four_directions[direction_indexes]
        # print(directions.shape)

        move_ranges = np.zeros((B, N)) + a / 2

        '''
        constrain the moving ranges to avoid collision
        '''
        for i in range(N):
            for j in range(N):
                if not i == j:
                    # i --> right
                    the_idx = (direction_indexes[:, i] == 0) & (x[:, j] > x[:, i]) & (abs(y[:, i] - y[:, j]) <= a[:, 0])
                    cur_gap = abs(x[:, j] - x[:, i]) - a[:, 0]
                    move_ranges[the_idx, i] = np.minimum(move_ranges[the_idx, i], ((cur_gap - min_gap) / 2)[the_idx])

                    # i --> down
                    the_idx = (direction_indexes[:, i] == 1) & (y[:, j] < y[:, i]) & (abs(x[:, i] - x[:, j]) <= a[:, 0])
                    cur_gap = abs(y[:, j] - y[:, i]) - a[:, 0]
                    move_ranges[the_idx, i] = np.minimum(move_ranges[the_idx, i], ((cur_gap - min_gap) / 2)[the_idx])

                    # i --> left
                    the_idx = (direction_indexes[:, i] == 2) & (x[:, j] < x[:, i]) & (abs(y[:, i] - y[:, j]) <= a[:, 0])
                    cur_gap = abs(x[:, j] - x[:, i]) - a[:, 0]
                    move_ranges[the_idx, i] = np.minimum(move_ranges[the_idx, i], ((cur_gap - min_gap) / 2)[the_idx])

                    # i --> up
                    the_idx = (direction_indexes[:, i] == 3) & (y[:, j] > y[:, i]) & (abs(x[:, i] - x[:, j]) <= a[:, 0])
                    cur_gap = abs(y[:, j] - y[:, i]) - a[:, 0]
                    move_ranges[the_idx, i] = np.minimum(move_ranges[the_idx, i], ((cur_gap - min_gap) / 2)[the_idx])

        '''
        constrain on port
        '''
        for i in range(N):
            # i --> right
            if i < N - 1:
                the_idx = direction_indexes[:, i] == 0
                move_ranges[the_idx, i] = np.minimum(move_ranges[the_idx, i], (x[the_idx, N - 1] - x[the_idx, i]) / 2)

                that_idx = the_idx & (direction_indexes[:, N - 1] == 2)
                move_ranges[that_idx, N - 1] = np.minimum(move_ranges[that_idx, N - 1],
                                                          (x[that_idx, N - 1] - x[that_idx, i]) / 2)

            # i --> left
            if i > 0:
                the_idx = direction_indexes[:, i] == 2
                move_ranges[the_idx, i] = np.minimum(move_ranges[the_idx, i], (x[the_idx, i] - x[the_idx, 0]) / 2)

                that_idx = the_idx & (direction_indexes[:, 0] == 0)
                move_ranges[that_idx, 0] = np.minimum(move_ranges[that_idx, 0], (x[that_idx, i] - x[that_idx, 0]) / 2)

        '''
        intention suppression
        '''
        x_l, x_r = x, x + move_ranges * directions[:, :, 0]
        y_l, y_r = y, y + move_ranges * directions[:, :, 1]
        x_l, x_r = np.minimum(x_l, x_r), np.maximum(x_l, x_r)
        y_l, y_r = np.minimum(y_l, y_r), np.maximum(y_l, y_r)
        x_l, x_r = x_l - a / 2, x_r + a / 2
        y_l, y_r = y_l - a / 2, y_r + a / 2

        idx = np.arange(N)
        np.random.shuffle(idx)
        for i in idx:
            for j in idx:
                if not i == j:
                    the_idx = box_collision_batch(x_l[:, i], x_r[:, i], y_l[:, i], y_r[:, i],
                                                  x_l[:, j], x_r[:, j], y_l[:, j], y_r[:, j], )

                    # do suppress
                    for b in range(B):
                        if the_idx[b]:
                            i_supp = [i, j][np.random.randint(2)]
                            move_ranges[b, i_supp] = 0
                            x_l[b, i_supp], x_r[b, i_supp] = x[b, i_supp] - a[b, i_supp] / 2, x[b, i_supp] + a[
                                b, i_supp] / 2
                            y_l[b, i_supp], y_r[b, i_supp] = y[b, i_supp] - a[b, i_supp] / 2, y[b, i_supp] + a[
                                b, i_supp] / 2

        '''
        do parameterization
        '''
        self.xy_parameter = nn.Parameter(torch.zeros(move_ranges.shape).cuda())

        u_side = u_info[:, :, :4]
        u_shift = u_info[:, :, 4:]
        u_shift_val = u_shift[u_side > 0]

        u_shift_val = np.clip(u_shift_val * 8, a_min=-1 + 1e-3, a_max=1 - 1e-3)
        u_shift_val_atanh = atanh(u_shift_val)

        self.u_parameter = nn.Parameter(torch.tensor(u_shift_val_atanh).to(torch.float).cuda())

        self.raw = raw
        self.raw_xy = to_tensor(raw[:, :, :2])
        self.raw_a = to_tensor(raw[:, :, 2:3])
        self.u_info = to_tensor(u_info)
        self.parameter_factor = to_tensor(move_ranges)
        self.parameter_directions = to_tensor(directions)
        self.optimizer = optim.Adam([self.xy_parameter, self.u_parameter], lr=self.lr)

    @staticmethod
    def non_linear(x):
        return (torch.exp(x) - torch.exp(-x)) / (torch.exp(x) + 1e6 * torch.exp(-x))

    def to_para(self):

        """xy"""
        delta = self.non_linear(self.xy_parameter) * self.parameter_factor
        delta_xy = self.parameter_directions * delta[:, :, None]
        xy = self.raw_xy + delta_xy

        """u"""
        u_shift_val = torch.tanh(self.u_parameter) / 8

        u_side = self.u_info[:, :, :4]
        u_shift = torch.zeros(u_side.shape).to(torch.float).cuda()
        u_shift[u_side > 0] = u_shift_val

        u_info = torch.cat([u_side, u_shift], 2)

        return xy, self.raw_a, u_info

    def to_raw_final(self):
        from circuit import get_gap
        xy, a, u_info = self.to_para()
        xy = to_np(xy)
        a = to_np(a)[:, :, 0]
        u_info = to_np(u_info)

        x, y, w, u = xy[:, :, 0], xy[:, :, 1], self.w, get_u(u_info)

        raw = np.zeros((x.shape[0], x.shape[1], 9), dtype=np.float32)

        raw[:, :, 0] = x
        raw[:, :, 1] = y
        raw[:, :, 2] = a
        raw[:, :, 3] = w
        raw[:, :, -1] = u

        for i in range(raw.shape[0]):
            for j in range(raw.shape[1]):
                # print('u x y a w', u[i, j], x[i, j], y[i, j], a[i, j], w[i, j])
                while u[i, j] > 1: u[i, j] -= 1
                while u[i, j] < 0: u[i, j] += 1
                slit_info = np.array(get_gap(u[i, j], x[i, j], y[i, j], a[i, j], w[i, j]))
                raw[i, j, 4:8] = slit_info
        return raw

    def main(self, vis=False):
        iter_step = 20
        num_iter = 10
        total_step = iter_step * num_iter

        for it in range(num_iter):
            if (it + 1) % 5 == 0:
                self.lr *= 0.3
            for step in range(iter_step):
                node_attr, edge_attr, adj = prepare_input_batch(self.to_para())

                pred = model(input=(node_attr, edge_attr, adj))
                pred_n = torch.sqrt(pred[:, 0, :] ** 2 + pred[:, 1, :] ** 2)
                loss = self.obj(pred_n)
                tmp_loss = to_np(loss)
                loss = loss.mean()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if (step + 1) % 20 == 0 and vis:
                    print(f'Iter {it}/{num_iter} step {step + 1}/{iter_step}  loss {loss.item()}')

            optimized_raw = self.to_raw_final()
            self.parameterize(raw=optimized_raw)

        optimized_raw = self.to_raw_final()
        i = np.argmin(tmp_loss)
        tag = int(time.time()) % 1000000

        our_para = optimized_raw[i]
        our_pred = to_np(pred[i])

        our_band, our_H = get_pass_band(our_pred)

        IOU = get_iou(self.target_band, our_band)

        if vis:
            def plot_label_db(ax, pred):
                tmp = np.sqrt(pred[1] ** 2 + pred[0] ** 2)
                tmp = to_db(tmp)
                ax.plot(np.linspace(200, 400, 5001), tmp, color='darkgreen', alpha=0.8, label='Model Prediced S2P')
                ax.set_xlim([200, 400])
                ax.set_ylim([-40, 0])
                ax.set_yticks([-6, -20])

            fig, ax = plt.subplots(1, 2, sharex='col', sharey='col', dpi=100, figsize=(12, 6))
            plot_circuit(ax[0], our_para)
            plot_label_db(ax[1], our_pred)
            plot_band(ax[1], band=self.target_band)
            plot_pred_band(ax[1], our_band)
            plt.legend()
            ax[0].set_title('Circuit')
            ax[1].set_title(f'Target band [{self.target_band[0]},{self.target_band[1]}], Delivered band [{our_band[0]:.0f},{our_band[1]:.0f}]\nIOU: {IOU*100:.1f} Insertion Loss: {-our_H:.1f}db')
            plt.show()
            plt.close()

        return IOU, our_H, our_para


# ======================================================================================================================

# dataset = PreparedDataset(num_block=4, train_ratio=1.0)


def solve(target_band,
          alpha, beta, cutoff_width,
          vis):
    the_num = 4
    the_type = 2

    generator = CircuitGenerator(num_resonator=the_num, a_range=[50, 90])
    raw = [generator.sample(tp=the_type)[0] for _ in range(2000)]
    raw = np.array(raw).astype(np.float32)

    solver = InverseDesigner()

    solver.set_objective(
        obj=construct_objective(band=target_band, cutoff_width=cutoff_width, alpha=alpha, beta=beta)
    )
    solver.target_band = target_band

    solver.parameterize(raw)
    return solver.main(vis=vis)


# ======================================================================================================================
def load_band(txt):
    l = open(txt, 'r').readlines()[-1]
    l = l.strip().split()
    ctr = float(l[0])
    bw = float(l[1])
    return ctr - bw / 2, ctr + bw / 2


def random_float(l, r):
    return np.random.rand() * (r - l) + l


if __name__ == '__main__':
    from progressbar import ProgressBar

    band_list = [(260, 290)]

    bar = ProgressBar()
    IOU_list = []
    H_list = []
    for i in bar(range(len(band_list))):
        IOU, H, para = solve(target_band=band_list[i],
                             cutoff_width=random_float(0.5, 0.8),
                             alpha=random_float(0.5, 0.8),
                             beta=random_float(0.5, 0.8),
                             vis=True)
