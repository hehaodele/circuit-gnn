import os
import numpy as np
from easydict import EasyDict
from utils import pickle_load, pickle_save
from torch.utils.data import Dataset

STATS = EasyDict()

STATS.node_attr = np.array([
    [0.0, 1.0], [0.0, 1.0], [75, 14.5],
    [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 0.07], [0.0, 0.07], [0.0, 0.07], [0.0, 0.07],

]).astype(np.float32)

STATS.edge_attr = np.array([
    [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 0.07], [0.0, 0.07], [0.0, 0.07], [0.0, 0.07],
    [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 0.07], [0.0, 0.07], [0.0, 0.07], [0.0, 0.07],
    [0.0, 65.0],
    [0.0, 65.0],
    [-1.60, 1.550],
    [0.320, 0.400],
]).astype(np.float32)


def normalize(x, stat, inverse=False):
    tmp_shape = x.shape
    tmp_len = x.shape[-1]
    if not inverse:
        return ((x.reshape(-1, tmp_len) - stat[:, 0][None, :]) / stat[:, 1][None, :]).reshape(tmp_shape)
    else:
        return (x.reshape(-1, tmp_len) * stat[:, 1][None, :] + stat[:, 0][None, :]).reshape(tmp_shape)


def load_data_list(root='./data', list_name='train_list.txt'):
    data_list = [x.strip() for x in open(root + '/' + list_name, 'r').readlines()]
    return data_list


class CircuitDataset(Dataset):

    def __init__(self,
                 data_root,
                 num_block,
                 data_list=None,
                 return_fn=False,
                 circuit_type=-1):
        self.n = num_block
        self.data_folder = data_root + '/data'
        if data_list is None:
            data_list = []
            for fd in filter(lambda x: x.startswith('num'), os.listdir(self.data_folder)):
                data_list += [fd + '/' + fn for fn in os.listdir(self.data_folder + '/' + fd)]

        self.data_list = [*filter(lambda x: x.startswith(f'num{num_block}'), data_list)]
        if circuit_type > -1:
            self.data_list = [*filter(lambda x: x.startswith(f'num{num_block}_type{circuit_type}'), self.data_list)]

        self.data_root = data_root
        self.return_fn = return_fn

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, i):
        fn = self.data_list[i]
        fn = os.path.join(self.data_folder, fn)

        tmp = pickle_load(fn)
        label, raw = tmp['label'], tmp['raw']
        node_attr, edge_attr, adj = tmp['node'], tmp['edge'], tmp['adj']

        node_attr = normalize(node_attr, STATS.node_attr)
        edge_attr = normalize(edge_attr, STATS.edge_attr)

        if self.return_fn:
            return (node_attr, edge_attr, adj), label.astype(np.float32), raw.astype(np.float32), fn

        return (node_attr, edge_attr, adj), label.astype(np.float32), raw.astype(np.float32)

    # def save(self, path):
    #     pickle_save(path, {
    #         'n': self.n,
    #         'fd': self.fd,
    #         'fn': self.fn,
    #         'idx': self.idx
    #     })
    #
    # def load(self, path):
    #     tmp = pickle_load(path)
    #     self.n = tmp['n']
    #     self.fn = tmp['fn']
    #     self.fd = tmp['fd']
    #     self.idx = tmp['idx']
    #
    # def collect_raw(self):
    #     raw_all = []
    #     for i, j in self.idx:
    #         fd, fn = self.fd[i], self.fn[i][j]
    #         fn = os.path.join(self.data_folder, fd, fn)
    #         raw = pickle_load(fn)['raw']
    #         raw_all.append(raw)
    #     raw_all = np.array(raw_all).astype(np.float32)
    #     return raw_all


if __name__ == '__main__':
    dataset = CircuitDataset(data_root='./data', data_list=load_data_list('./data', 'test_list.txt'), num_block=3)
    print(len(dataset))
    print(dataset[0])
