import numpy as np
from easydict import EasyDict
import random

opt = EasyDict()

'''
the range of the side length of the block resonator
Please fill the range here.
'''
opt.a_range = [50, 100]

'''
sep: from 1/80 a  to 1/5 a
'''
opt.gap_min_ratio = 1.0 / 80
opt.gap_max_ratio = 1.0 / 5

opt.gap_min_ratio_tight = 1.0 / 50
opt.gap_max_ratio_tight = 1.0 / 20

'''
shift: three level of shift [0.5, 0.2, 0.0] * a'''
opt.shift_range = [0.5, 0.2, 0.0]

opt.w = 6
opt.gap_len = 2.4

DATA_CONST = opt


# ======================================================================================================================
def log_uniform(l, r):
    l, r = np.log(l), np.log(r)
    return np.exp((r - l) * np.random.rand() + l)


def uniform(l, r):
    return (r - l) * np.random.rand() + l


def no_collison(xy, a):
    xy = np.array(xy).reshape(-1, 2)
    n = len(xy)
    for i in range(n):
        for j in range(i):
            dx, dy = np.abs(xy[i] - xy[j])
            if dx < a and dy < a:
                return False
    return True


# ======================================================================================================================
def add_left(x, y, a, sft_range):
    return x - a - a * log_uniform(opt.gap_min_ratio, opt.gap_max_ratio), \
           y + uniform(-sft_range, sft_range)


def add_right(x, y, a, sft_range, tight=False):
    if not tight:
        gap = log_uniform(opt.gap_min_ratio, opt.gap_max_ratio)
    else:
        gap = log_uniform(opt.gap_min_ratio_tight, opt.gap_max_ratio_tight)
    return x + a + a * gap, \
           y + uniform(-sft_range, sft_range)


def add_up(x, y, a, sft_range):
    return x + uniform(-sft_range, sft_range), \
           y + a + a * log_uniform(opt.gap_min_ratio, opt.gap_max_ratio)


def add_down(x, y, a, sft_range):
    return x + uniform(-sft_range, sft_range), \
           y - a - a * log_uniform(opt.gap_min_ratio, opt.gap_max_ratio)


# ======================================================================================================================
def get_gap(u, x, y, a, w, glen=opt.gap_len):
    """
    :param glen: length of the gap
    :param u: gap position theta = 2pi * u (u is from 0 to 1)
    :param a: block side length
    :param w: block side width
    :return:
    """
    if 0 <= u < 1.0 / 8 or 7.0 / 8 <= u <= 1:  # left open
        xg = x + a / 2 - w / 2
        yg = y + np.tan(u * 2 * np.pi) * (a / 2 - w)
        dx = w
        dy = glen

    if 1.0 / 8 <= u < 3.0 / 8:  # up open
        xg = x - np.tan((u - 1.0 / 4) * 2 * np.pi) * (a / 2 - w)
        yg = y + a / 2 - w / 2
        dx = glen
        dy = w

    if 3.0 / 8 <= u < 5.0 / 8:  # left open
        xg = x - a / 2 + w / 2
        yg = y - np.tan((u - 1.0 / 2) * 2 * np.pi) * (a / 2 - w)
        dx = w
        dy = glen

    if 5.0 / 8 <= u < 7.0 / 8:  # down open
        xg = x + np.tan((u - 3.0 / 4) * 2 * np.pi) * (a / 2 - w)
        yg = y - a / 2 + w / 2
        dx = glen
        dy = w

    return np.array([xg, yg, dx, dy]).astype(np.float32)


def no_collison(xy, a):
    n = len(xy)
    for i in range(n):
        for j in range(i):
            dx, dy = np.abs(xy[i] - xy[j])
            if dx < a and dy < a:
                return False
    return True


def no_board_exceed(xy, a):
    x = xy[:, 0]
    return x[0] == np.min(x) and x[-1] == np.max(x)


# ======================================================================================================================
from random import randint


class SquareGenrator(object):

    def __init__(self, num):
        self.generators = {
            2: self.gen_2,
            3: self.gen_3,
            4: self.gen_4,
            5: self.gen_5,
            6: self.gen_6,
            7: None,
            8: self.gen_8,
        }
        self.sample = self.generators[num]

    @staticmethod
    def gen_2(a):
        x0, y0 = 0, 0
        x1, y1 = add_right(x0, y0, a, sft_range=a * 0.5)
        xy = [[x0, y0], [x1, y1]]
        xy = np.array(xy).reshape(-1, 2)
        return xy, 0

    @staticmethod
    def gen_3(a, tp=-1):
        if tp > -1:
            t = tp
        else:
            t = randint(0, 3)
        while True:
            if t == 0:
                x0, y0 = 0, 0
                x1, y1 = add_right(x0, y0, a, a * 0.5)
                x2, y2 = add_right(x1, y1, a, a * 0.5)

            if t == 1:
                x0, y0 = 0, 0
                x2, y2 = add_right(x0, y0, a, a * 0.5)
                x1, y1 = add_down(x0, y0, a, a * 0.5)

            if t == 2:
                x0, y0 = 0, 0
                x2, y2 = add_right(x0, y0, a, a * 0.0)
                x1, y1 = add_down((x0 + x2) / 2, y0, a, a * 0.2)

            if t == 3:
                x0, y0 = 0, 0
                x1, y1 = add_right(x0, y0, a, a * 0.5)
                x2, y2 = add_down(x1, y1, a, a * 0.5)

            xy = [x0, y0, x1, y1, x2, y2]
            xy = np.array(xy).reshape(-1, 2)
            if no_collison(xy, a * (1 + opt.gap_min_ratio)) and no_board_exceed(xy, a):
                break
        return xy, t

    @staticmethod
    def gen_4(a, tp=-1):
        if isinstance(tp, list):
            t = random.choice(tp)
        elif tp > -1:
            t = tp
        else:
            t = randint(0, 11)
        while True:
            if t == 0:
                x0, y0 = 0, 0
                x1, y1 = add_right(x0, y0, a, a * 0.5)
                x2, y2 = add_right(x1, y1, a, a * 0.5)
                x3, y3 = add_right(x2, y2, a, a * 0.5)

            if t == 1:
                x0, y0 = 0, 0
                x3, y3 = add_right(x0, y0, a, a * 0.5)
                x1, y1 = add_down(x0, y0, a, a * 0.5)
                x2, y2 = add_down(x3, y3, a, a * 0.5)

            if t == 2:
                x0, y0 = 0, 0
                x3, y3 = add_right(x0, y0, a, a * 0.0)
                x1, y1 = add_down(x0, y0, a, a * 0.0)
                x2, y2 = add_down(x3, y3, a, a * 0.0)
                resample_gap = a * log_uniform(opt.gap_min_ratio, opt.gap_max_ratio)
                curr_gap = x2 - x1 - a
                x2 += (resample_gap - curr_gap) / 2
                x1 -= (resample_gap - curr_gap) / 2

            if t == 3:
                x0, y0 = 0, 0
                x3, y3 = add_right(x0, y0, a, a * 0.5)
                x1, y1 = add_down(x0, y0, a, a * 0.5)
                x2, y2 = add_up(x0, y0, a, a * 0.5)

            if t == 4:
                x0, y0 = 0, 0
                x3, y3 = add_right(x0, y0, a, a * 0.5)
                x1, y1 = add_down(x0, y0, a, a * 0.5)
                x2, y2 = add_up(x3, y3, a, a * 0.5)

            if t == 5:
                x0, y0 = 0, 0
                x3, y3 = add_right(x0, y0, a, a * 0.0)
                x1, y1 = add_down((x0 + x3) / 2, y0, a, a * 0.2)
                x2, y2 = add_up((x0 + x3) / 2, y0, a, a * 0.2)

            if t == 6:
                x0, y0 = 0, 0
                x1, y1 = add_down(x0, y0, a, a * 0.0)
                x2, y2 = add_right(x0, (y0 + y1) / 2, a, a * 0.2)
                x3, y3 = add_right(x2, y2, a, a * 0.5)

            if t == 7:
                x0, y0 = 0, 0
                x1, y1 = add_right(x0, y0, a, a * 0.5)
                x3, y3 = add_right(x1, y1, a, a * 0.5)
                x2, y2 = add_down(x0, y0, a, a * 0.5)

            if t == 8:
                x0, y0 = 0, 0
                x1, y1 = add_right(x0, y0, a, a * 0.5)
                x3, y3 = add_right(x1, y1, a, a * 0.5)
                x2, y2 = add_down(x1, y1, a, a * 0.5)

            if t == 9:
                x0, y0 = 0, 0
                x1, y1 = add_right(x0, y0, a, a * 0.0)
                x3, y3 = add_right(x1, y1, a, a * 0.5)
                x2, y2 = add_down((x0 + x1) / 2, y0, a, a * 0.2)

            if t == 10:
                x1, y1 = 0, 0
                x2, y2 = add_down(x1, y1, a, a * 0.0)
                x0, y0 = add_right(x1, (y1 + y2) / 2, a, a * 0.2)
                x0 = - x0
                x3, y3 = add_right(x1, (y1 + y2) / 2, a, a * 0.2)

            if t == 11:
                x0, y0 = 0, 0
                x1, y1 = add_right(x0, y0, a, a * 0.5)
                x2, y2 = add_down(x1, y1, a, a * 0.5)
                x3, y3 = add_right(x2, y2, a, a * 0.5)

            """ New added after ICML
            """
            if t == 12:
                x0, y0 = 0, 0
                x2, y2 = add_right(x0, y0, a, a * 0.0)
                x1, y1 = add_down(x0, y0, a, a * 0.1)
                x3, y3 = add_down(x2, y2, a, a * 0.1)

            if t == 13:
                x0, y0 = 0, 0
                x1, y1 = add_right(x0, y0, a, a * 0.0)
                x2, y2 = add_down(x1, y1, a, a * 0.0)
                x3, y3 = add_right(x2, y2, a, a * 0.0)

            if t == 14:
                x0, y0 = 0, 0
                x2, y2 = add_right(x0, y0, a, a * 0.0)
                x1, y1 = add_down((x0 + x2) / 2, y0, a, a * 0.2)
                x3, y3 = add_right(x1, y1, a, a * 0.0)

            xy = [x0, y0, x1, y1, x2, y2, x3, y3]
            # print(xy)
            xy = np.array(xy).reshape(-1, 2)
            if no_collison(xy, a * (1 + opt.gap_min_ratio)) and no_board_exceed(xy, a):
                break
        return xy, t

    @staticmethod
    def gen_5(a, tp=-1):
        if isinstance(tp, list):
            t = random.choice(tp)
        elif tp > -1:
            t = tp
        else:
            t = randint(0, 8)
        while True:
            if t == 0:
                x0, y0 = 0, 0
                x4, y4 = add_right(x0, y0, a, a * 0.5)
                x1, y1 = add_up(x0, y0, a, a * 0.5)
                x2, y2 = add_down(x0, y0, a, a * 0.5)
                x3, y3 = add_down(x4, y4, a, a * 0.5)

            if t == 1:
                x0, y0 = 0, 0
                x4, y4 = add_right(x0, y0, a, a * 0.0)
                x1, y1 = add_up((x0 + x4) / 2, y0, a, a * 0.2)
                x2, y2 = add_down(x0, y0, a, a * 0.5)
                x3, y3 = add_down(x4, y4, a, a * 0.5)

            if t == 2:
                x0, y0 = 0, 0
                x4, y4 = add_right(x0, y0, a, a * 0.0)
                x2, y2 = add_down(x0, y0, a, a * 0.0)
                x3, y3 = add_down(x4, y4, a, a * 0.0)
                resample_gap = a * log_uniform(opt.gap_min_ratio, opt.gap_max_ratio)
                curr_gap = x3 - x2 - a
                x3 += (resample_gap - curr_gap) / 2
                x2 -= (resample_gap - curr_gap) / 2
                x1, y1 = add_up(x0, y0, a, a * 0.5)

            if t == 3:
                x0, y0 = 0, 0
                x4, y4 = add_right(x0, y0, a, a * 0.0)
                x2, y2 = add_down(x0, y0, a, a * 0.0)
                x3, y3 = add_down(x4, y4, a, a * 0.0)
                resample_gap = a * log_uniform(opt.gap_min_ratio, opt.gap_max_ratio)
                curr_gap = x3 - x2 - a
                x3 += (resample_gap - curr_gap) / 2
                x2 -= (resample_gap - curr_gap) / 2
                x1, y1 = add_up((x0 + x4) / 2, y0, a, a * 0.2)

            if t == 4:
                x0, y0 = 0, 0
                x2, y2 = add_right(x0, y0, a, a * 0.5)
                x4, y4 = add_right(x2, y2, a, a * 0.5)
                x1, y1 = add_down(x0, y0, a, a * 0.5)
                x3, y3 = add_down(x4, y4, a, a * 0.5)

            if t == 5:
                x0, y0 = 0, 0
                x2, y2 = add_right(x0, y0, a, a * 0.5)
                x4, y4 = add_right(x2, y2, a, a * 0.5)
                x1, y1 = add_up(x2, y2, a, a * 0.5)
                x3, y3 = add_down(x2, y2, a, a * 0.5)

            if t == 6:
                x0, y0 = 0, 0
                x2, y2 = add_right(x0, y0, a, a * 0.0)
                x4, y4 = add_right(x2, y2, a, a * 0.0)
                x1, y1 = add_down((x0 + x2) / 2, y0, a, a * 0.2)
                x3, y3 = add_down((x2 + x4) / 2, y4, a, a * 0.2)
                resample_gap = a * log_uniform(opt.gap_min_ratio, opt.gap_max_ratio)
                curr_gap = x3 - x1 - a
                x3 += (resample_gap - curr_gap) / 2
                x1 -= (resample_gap - curr_gap) / 2

            if t == 7:
                x0, y0 = 0, 0
                x2, y2 = add_right(x0, y0, a, a * 0.2)
                x4, y4 = add_right(x2, y2, a, a * 0.2)
                x1, y1 = add_down(x0, y0, a, a * 0.0)
                x3, y3 = add_down(x4, y4, a, a * 0.0)
                y2 -= a * 0.5

            if t == 8:
                x0, y0 = 0, 0
                x2, y2 = add_right(x0, y0, a, a * 0.2)
                y2 += a * 0.5
                x1, y1 = add_down(x2, y2, a, a * 0.0)
                x4, y4 = add_right(x2, y2, a, a * 0.2)
                x3, y3 = add_down(x4, y4, a, a * 0.0)

            """ New
            """
            if t == 9:
                x0, y0 = 0, 0
                x2, y2 = add_right(x0, y0, a, a * 0.0)
                x4, y4 = add_right(x2, y2, a, a * 0.0)
                x1, y1 = add_down(x0, y0, a, a * 0.0)
                x3, y3 = add_down(x2, y2, a, a * 0.0)

            if t == 10:
                x0, y0 = 0, 0
                x2, y2 = add_right(x0, y0, a, a * 0.0)
                x4, y4 = add_right(x2, y2, a, a * 0.0)
                x1, y1 = add_down(x0, y0, a, a * 0.0)
                x3, y3 = add_down(x4, y4, a, a * 0.0)

            if t == 11:
                x0, y0 = 0, 0
                x1, y1 = add_right(x0, y0, a, a * 0.0)
                x2, y2 = add_down(x0, y0, a, a * 0.0)
                x3, y3 = add_down(x1, y1, a, a * 0.0)
                x4, y4 = add_right(x3, y3, a, a * 0.0)

            if t == 12:
                x0, y0 = 0, 0
                x1, y1 = add_right(x0, y0, a, a * 0.0, tight=True)
                x2, y2 = add_right(x1, y1, a, a * 0.0, tight=True)
                x4, y4 = add_right(x2, y2, a, a * 0.0, tight=True)
                x3, y3 = add_down(x0, y0, a, a * 0.0)

            if t == 13:
                x0, y0 = 0, 0
                x1, y1 = add_right(x0, y0, a, a * 0.0, tight=True)
                x2, y2 = add_right(x1, y1, a, a * 0.0, tight=True)
                x4, y4 = add_right(x2, y2, a, a * 0.0, tight=True)
                x3, y3 = add_down((x0 + x1) / 2, (y0 + y1) / 2, a, a * 0.0)

            if t == 14:
                x0, y0 = 0, 0
                x1, y1 = add_right(x0, y0, a, a * 0.0, tight=True)
                x2, y2 = add_right(x1, y1, a, a * 0.0, tight=True)
                x4, y4 = add_right(x2, y2, a, a * 0.0, tight=True)
                x3, y3 = add_down(x1, y1, a, a * 0.0)

            xy = [x0, y0, x1, y1, x2, y2, x3, y3, x4, y4]
            xy = np.array(xy).reshape(-1, 2)
            if no_collison(xy, a * (1 + opt.gap_min_ratio)) and no_board_exceed(xy, a):
                break
        return xy, t

    @staticmethod
    def gen_6(a, tp=-1):
        if isinstance(tp, list):
            t = random.choice(tp)
        elif tp > -1:
            t = tp
        else:
            t = randint(0, 5)
        while True:
            if t == 0:
                x0, y0 = 0, 0
                x1, y1 = add_right(x0, y0, a, a * 0.5)
                x3, y3 = add_right(x1, y1, a, a * 0.0)
                x5, y5 = add_right(x3, y3, a, a * 0.5)

                x2, y2 = add_down(x1, y1, a, a * 0.0)
                x4, y4 = add_down(x3, y3, a, a * 0.0)
                resample_gap = a * log_uniform(opt.gap_min_ratio, opt.gap_max_ratio)
                curr_gap = x4 - x2 - a
                x4 += (resample_gap - curr_gap) / 2
                x2 -= (resample_gap - curr_gap) / 2

            if t == 1:
                x0, y0 = 0, 0
                x1, y1 = add_right(x0, y0, a, a * 0.0)
                x3, y3 = add_right(x1, y1, a, a * 0.5)
                x5, y5 = add_right(x3, y3, a, a * 0.0)

                x2, y2 = add_down((x0 + x1) / 2, y1, a, a * 0.2)
                x4, y4 = add_down((x3 + x5) / 2, y3, a, a * 0.2)

            if t == 2:
                x1, y1 = 0, 0
                x2, y2 = add_down(x1, y1, a, a * 0.0)
                x0, y0 = add_left(x1, (y1 + y2) / 2, a, a * 0.2)
                x3, y3 = add_right(x1, y1, a, a * 0.2)
                x4, y4 = add_down(x3, y3, a, a * 0.0)
                x5, y5 = add_right(x3, (y3 + y4) / 2, a, a * 0.2)

            if t == 3:
                x1, y1 = 0, 0
                x2, y2 = add_down(x1, y1, a, a * 0.0)
                x0, y0 = add_left(x1, (y1 + y2) / 2, a, a * 0.2)
                x4, y4 = add_right(x1, (y1 + y2) / 2, a, a * 0.2)
                x3, y3 = add_up(x4, y4, a, a * 0.0)
                x5, y5 = add_right(x3, (y3 + y4) / 2, a, a * 0.2)

            if t == 4:
                x0, y0 = 0, 0
                x5, y5 = add_right(x0, y0, a, a * 0.0)

                x1, y1 = add_down(x0, y0, a, a * 0.0)
                x3, y3 = add_down(x5, y5, a, a * 0.0)
                resample_gap = a * log_uniform(opt.gap_min_ratio, opt.gap_max_ratio)
                curr_gap = x3 - x1 - a
                x3 += (resample_gap - curr_gap) / 2
                x1 -= (resample_gap - curr_gap) / 2

                x2, y2 = add_up(x0, y0, a, a * 0.0)
                x4, y4 = add_up(x5, y5, a, a * 0.0)
                resample_gap = a * log_uniform(opt.gap_min_ratio, opt.gap_max_ratio)
                curr_gap = x4 - x2 - a
                x4 += (resample_gap - curr_gap) / 2
                x2 -= (resample_gap - curr_gap) / 2

            if t == 5:
                x0, y0 = 0, 0
                x2, y2 = add_right(x0, y0, a, a * 0.2)
                x5, y5 = add_right(x2, y2, a, a * 0.2)
                x1, y1 = add_down(x0, y0, a, a * 0.1)
                x3, y3 = add_down(x2, y2, a, a * 0.1)
                x4, y4 = add_down(x5, y5, a, a * 0.1)

            """ After ICML
            """

            if t == 6:
                x0, y0 = 0, 0
                x1, y1 = add_right(x0, y0, a, a * 0.0, tight=True)
                x4, y4 = add_right(x1, y1, a, a * 0.0, tight=True)
                x5, y5 = add_right(x4, y4, a, a * 0.0, tight=True)
                x2, y2 = add_up(x0, y0, a, a * 0.0)
                x3, y3 = add_down(x0, y0, a, a * 0.0)

            if t == 7:
                x0, y0 = 0, 0
                x1, y1 = add_right(x0, y0, a, a * 0.0, tight=True)
                x4, y4 = add_right(x1, y1, a, a * 0.0, tight=True)
                x5, y5 = add_right(x4, y4, a, a * 0.0, tight=True)
                x2, y2 = add_up(x1, y1, a, a * 0.0)
                x3, y3 = add_down(x1, y1, a, a * 0.0)

            if t == 8:
                x0, y0 = 0, 0
                x1, y1 = add_right(x0, y0, a, a * 0.0, tight=True)
                x4, y4 = add_right(x1, y1, a, a * 0.0, tight=True)
                x5, y5 = add_right(x4, y4, a, a * 0.0, tight=True)
                x2, y2 = add_up(x0, y0, a, a * 0.0)
                x3, y3 = add_down(x1, y1, a, a * 0.0)

            if t == 9:
                x0, y0 = 0, 0
                x1, y1 = add_right(x0, y0, a, a * 0.0, tight=True)
                x4, y4 = add_right(x1, y1, a, a * 0.0, tight=True)
                x5, y5 = add_right(x4, y4, a, a * 0.0, tight=True)
                x2, y2 = add_up(x1, y1, a, a * 0.0)
                x3, y3 = add_down(x4, y4, a, a * 0.0)

            xy = [x0, y0, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5]
            xy = np.array(xy).reshape(-1, 2)
            if no_collison(xy, a * (1 + opt.gap_min_ratio)) and no_board_exceed(xy, a):
                break
        return xy, t

    @staticmethod
    def gen_8(a, tp=-1):
        if tp > -1:
            t = tp
        else:
            t = randint(0, 1)
        while True:
            if t == 0:
                x0, y0 = 0, 0
                x1, y1 = add_right(x0, y0, a, a * 0.1)
                x4, y4 = add_right(x1, y1, a, a * 0.0)
                x7, y7 = add_right(x4, y4, a, a * 0.1)

                x2, y2 = add_down(x1, y1, a, a * 0.0)
                x5, y5 = add_down(x4, y4, a, a * 0.0)
                resample_gap = a * log_uniform(opt.gap_min_ratio, opt.gap_max_ratio)
                curr_gap = x5 - x2 - a
                x5 += (resample_gap - curr_gap) / 2
                x2 -= (resample_gap - curr_gap) / 2

                x3, y3 = add_up(x1, y1, a, a * 0.0)
                x6, y6 = add_up(x4, y4, a, a * 0.0)
                resample_gap = a * log_uniform(opt.gap_min_ratio, opt.gap_max_ratio)
                curr_gap = x6 - x3 - a
                x6 += (resample_gap - curr_gap) / 2
                x3 -= (resample_gap - curr_gap) / 2

            if t == 1:
                x0, y0 = 0, 0
                x1, y1 = add_right(x0, y0, a, a * 0.1)
                x4, y4 = add_right(x1, y1, a, a * 0.0)
                x7, y7 = add_right(x4, y4, a, a * 0.1)

                x2, y2 = add_down(x1, y1, a, a * 0.0)
                x5, y5 = add_down(x4, y4, a, a * 0.0)
                resample_gap = a * log_uniform(opt.gap_min_ratio, opt.gap_max_ratio)
                curr_gap = x5 - x2 - a
                x5 += (resample_gap - curr_gap) / 2
                x2 -= (resample_gap - curr_gap) / 2

                x3, y3 = add_up(x0, y0, a, a * 0.2)
                x6, y6 = add_up(x7, y7, a, a * 0.2)

            xy = [x0, y0, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7]
            xy = np.array(xy).reshape(-1, 2)
            if no_collison(xy, a * (1 + opt.gap_min_ratio)) and no_board_exceed(xy, a):
                break
        return xy, t


# ======================================================================================================================

class CircuitGenerator(object):
    def __init__(self,
                 num_resonator,
                 a_range=None):
        self.num_resonator = num_resonator
        self.square_generator = SquareGenrator(num=num_resonator)
        if a_range is None:
            self.a_range = opt.a_range
        else:
            self.a_range = a_range

    def sample(self, tp=-1):
        a = uniform(self.a_range[0], self.a_range[1])
        w = opt.w

        xy, tp = self.square_generator.sample(a, tp)
        '''
        random up/down left/right flip
        '''
        if np.random.rand() < 0.5:  # left/right
            xy[:, 0] = - xy[:, 0]
            xy = np.flip(xy, 0)
        if np.random.rand() < 0.5:  # up/down
            xy[:, 1] = - xy[:, 1]
        xy = xy - xy[0, :]

        all_u = np.random.rand(self.num_resonator, 1)
        gap_info = [get_gap(u=u, x=x, y=y, a=a, w=w) for u, x, y in zip(all_u, xy[:, 0], xy[:, 1])]
        gap_info = np.array(gap_info).reshape(-1, 4)
        all_a = np.array([a] * self.num_resonator)[:, None]
        all_w = np.array([w] * self.num_resonator)[:, None]

        '''
        circuit_info: num x 9
        [x, y, a, w, gap_x, gap_y, gap_dx, gap_dy, u]
        '''

        circuit_info = np.concatenate(
            [xy, all_a, all_w, gap_info, all_u], 1
        )
        return circuit_info, tp


# ======================================================================================================================
def augment_u(para):
    assert len(para) == 4

    u = para[:, -1]
    x, y, a, w = para[:, 0], para[:, 1], para[:, 2], para[:, 3]

    circuits = []

    for i in range(4):
        for du in np.arange(0, 1, 1.0 / 20):
            new_u = u.copy()
            new_u[i] = (new_u[i] + du) % 1

            gap_info = [get_gap(_u, _x, _y, _a, _w) for _u, _x, _y, _a, _w in zip(new_u, x, y, a, w)]
            gap_info = np.array(gap_info).reshape(-1, 4)

            circuit_info = np.concatenate(
                [x[:, None], y[:, None], a[:, None], w[:, None], gap_info, new_u[:, None]], 1
            )

            circuits.append(circuit_info)

    return np.array(circuits).astype(np.float32)


# ======================================================================================================================
if __name__ == '__main__':
    from utils import plot_circuit
    import matplotlib.pyplot as plt
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_resonator', type=int, default=4, help="num of resonators from [3,4,5,6]")
    args = parser.parse_args()

    generator = CircuitGenerator(num_resonator=args.num_resonator)
    for i in range(100):
        f, ax = plt.subplots(1, 1, dpi=100, figsize=(8, 6))
        para, tp = generator.sample()
        plot_circuit(para=para, ax=ax)
        plt.title(f'{args.num_resonator}-Resonator (type {tp})')
        plt.show()
