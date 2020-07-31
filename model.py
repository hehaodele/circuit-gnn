import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader

# ======================================================================================================================
"""
Complex Prediction Network
"""


def complex_linear(layer, x):
    return (layer[0](x[0]) - layer[1](x[1]),
            layer[0](x[1]) - layer[1](x[0]))


def complex_apply(layer, x):
    return (layer[0](x[0]),
            layer[1](x[1]))


def complex_add(x, y):
    return (x[0] + y[0],
            x[1] + y[1])


class ComplexNet(nn.Module):
    def __init__(self,
                 nin=5,
                 nh=512,
                 dropout=0):
        super(ComplexNet, self).__init__()
        self.nh = nh
        self.n_input = nin

        self.fc1 = (nn.Linear(nin, nh), nn.Linear(nin, nh))
        self.fc2 = (nn.Linear(nh, nh), nn.Linear(nh, nh))
        self.fc22 = (nn.Linear(nh, nh), nn.Linear(nh, nh))
        self.fc2_bn = (nn.BatchNorm1d(nh), nn.BatchNorm1d(nh))
        self.fc22_bn = (nn.BatchNorm1d(nh), nn.BatchNorm1d(nh))
        self.fc3 = (nn.Linear(nh, nh), nn.Linear(nh, nh))
        self.fc32 = (nn.Linear(nh, nh), nn.Linear(nh, nh))
        self.fc3_bn = (nn.BatchNorm1d(nh), nn.BatchNorm1d(nh))
        self.fc32_bn = (nn.BatchNorm1d(nh), nn.BatchNorm1d(nh))
        self.fc4 = (nn.Linear(nh, 5001), nn.Linear(nh, 5001))

        self.fc1_r, self.fc1_i = self.fc1
        self.fc2_r, self.fc2_i = self.fc2
        self.fc22_r, self.fc22_i = self.fc22
        self.fc2_bn_r, self.fc2_bn_i = self.fc2_bn
        self.fc22_bn_r, self.fc22_bn_i = self.fc22_bn
        self.fc3_r, self.fc3_i = self.fc3
        self.fc32_r, self.fc32_i = self.fc32
        self.fc3_bn_r, self.fc3_bn_i = self.fc3_bn
        self.fc32_bn_r, self.fc32_bn_i = self.fc32_bn
        self.fc4_r, self.fc4_i = self.fc4

        if dropout > 0:
            self.dropout_2 = (nn.Dropout(dropout), nn.Dropout(dropout))
            self.dropout_3 = (nn.Dropout(dropout), nn.Dropout(dropout))
        else:
            self.dropout_2, self.dropout_3 = None, None

    def forward(self, x):
        x1 = complex_linear(self.fc1, x)
        x1 = (F.leaky_relu(x1[0], 0.2),
              F.leaky_relu(x1[1], 0.2))
        x = complex_apply((F.relu, F.relu),
                          complex_apply(self.fc2_bn,
                                        complex_linear(self.fc2, x1)))
        x2 = complex_apply((F.relu, F.relu),
                           complex_apply(self.fc22_bn,
                                         complex_add(x1,
                                                     complex_linear(self.fc22, x))))

        if self.dropout_2 is not None: complex_apply(self.dropout_2, x2)

        x = complex_apply((F.relu, F.relu),
                          complex_apply(self.fc3_bn,
                                        complex_linear(self.fc3, x2)))
        x3 = complex_apply((F.relu, F.relu),
                           complex_apply(self.fc32_bn,
                                         complex_add(x2,
                                                     complex_linear(self.fc32, x))))

        if self.dropout_3 is not None: complex_apply(self.dropout_3, x3)

        x = complex_linear(self.fc4, x3)

        return x


# ======================================================================================================================
'''
GNN
'''


class EdgeAttentionNet(nn.Module):
    def __init__(self,
                 n_edge_attr,
                 n_hid=32):
        super(EdgeAttentionNet, self).__init__()
        self.fc0 = nn.Linear(n_edge_attr, n_hid)
        self.fc1 = nn.Linear(n_hid, n_hid)
        self.fc2 = nn.Linear(n_hid, 1)

    def forward(self, x):
        x0 = self.fc0(x)
        x1 = self.fc1(F.relu(x0))
        x2 = self.fc2(F.relu(x0 + x1))
        return torch.sigmoid(x2)


class GraphInteractionLayer(nn.Module):

    def __init__(self,
                 n_node_attr,
                 n_node_code,
                 n_edge_attr,
                 n_edge_code, ):
        super(GraphInteractionLayer, self).__init__()

        self.edge_processor = nn.Linear(n_edge_attr + (n_node_attr + n_node_code) * 2, n_edge_code)
        self.node_processor = nn.Linear(n_node_attr + n_node_code + n_edge_code, n_node_code)

    def forward(self, node_code, node_attr, edge_attr, adj, return_edge_code=False):
        """
        :param return_edge_code: whether return [edge_code]
        :param node_code: B x N x D1
        :param node_attr: B x N x D2
        :param edge_attr: B x N x N x D3
        :param adj: B x N x N
        :return: new_node_code: B x N x D1
        """

        B, N = node_code.size(0), node_code.size(1)
        node_info = torch.cat([node_code, node_attr], 2)

        receiver_info = node_info[:, :, None, :].repeat(1, 1, N, 1)
        sender_info = node_info[:, None, :, :].repeat(1, N, 1, 1)

        edge_input = torch.cat([edge_attr, receiver_info, sender_info], 3)
        edge_code = F.leaky_relu(self.edge_processor(edge_input.reshape(B * N * N, -1)).reshape(B, N, N, -1))

        edge_agg = (edge_code * adj[:, :, :, None]).sum(2)

        node_input = torch.cat([node_info, edge_agg], 2)
        new_node_code = self.node_processor(node_input.reshape(B * N, -1)).reshape(B, N, -1)

        if return_edge_code: return new_node_code, edge_code

        return new_node_code


class GIN(nn.Module):
    """
    Graph Interaction Network
    """

    def __init__(self,
                 n_node_attr,
                 n_node_code,
                 n_edge_attr,
                 n_edge_code,
                 n_layers=2,
                 use_gpu=False,
                 dropout=0):
        super(GIN, self).__init__()

        self.layers = []
        for i in range(n_layers):
            layer = GraphInteractionLayer(n_node_attr=n_node_attr, n_node_code=n_node_code, n_edge_attr=n_edge_attr,
                                          n_edge_code=n_edge_code)
            self.layers.append(layer)
            setattr(self, 'gin_layer_{}'.format(i), layer)

        self.node_encoder = nn.Sequential(
            nn.Linear(n_node_attr, n_node_code),
            nn.LeakyReLU(0.1),
        )
        self.n_node_code = n_node_code
        self.n_layers = n_layers
        self.use_gpu = use_gpu

        if dropout > 0:
            self.drop_layers = [nn.Dropout(p=dropout)] * n_layers
        else:
            self.drop_layers = None

    def forward(self, node_attr, edge_attr, adj, return_edge_code=False):
        x = self.node_encoder(node_attr)
        x_edge_codes = []
        for i in range(self.n_layers):
            if return_edge_code:
                x, x_edge = self.layers[i](x, node_attr, edge_attr, adj, return_edge_code)
                x_edge_codes.append(x_edge)
            else:
                x = self.layers[i](x, node_attr, edge_attr, adj)
            x = F.leaky_relu(x)
            if self.drop_layers is not None:
                x = self.drop_layers[i](x)
        if return_edge_code:
            return x, x_edge_codes

        return x


# ======================================================================================================================
def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0)


class CircuitGNN(nn.Module):

    def __init__(self, args):
        super(CircuitGNN, self).__init__()

        nhid = args.len_hidden
        self.gnn_encoder = GIN(n_node_code=nhid, n_edge_code=nhid, n_node_attr=args.len_node_attr,
                               n_edge_attr=args.len_edge_attr, n_layers=args.gnn_layers,
                               use_gpu=args.use_gpu,
                               dropout=args.dropout)
        '''use cat summarization'''
        len_gnn_agg = nhid * 2
        self.predictor = ComplexNet(nin=len_gnn_agg, nh=args.len_hidden_predictor, dropout=args.dropout)
        self.apply(weights_init)

    def forward(self, input, return_edge_code=False):
        node_attr, edge_attr, adj = input
        if return_edge_code:
            gnn_node_codes, edge_codes = self.gnn_encoder(node_attr, edge_attr, adj, return_edge_code)
        else:
            gnn_node_codes = self.gnn_encoder(node_attr, edge_attr, adj)
        gnn_code = torch.cat([gnn_node_codes[:, 0, :], gnn_node_codes[:, -1, :]], 1)
        pred = self.predictor((gnn_code, gnn_code))
        pred = torch.cat([pred[0][:, None, :], pred[1][:, None, :]], 1)

        if return_edge_code:
            return torch.tanh(pred), edge_codes

        return torch.tanh(pred)


# ======================================================================================================================
class SimpleNN(nn.Module):

    def __init__(self, args, n_in):
        super(SimpleNN, self).__init__()
        self.predictor = ComplexNet(nin=n_in, nh=args.len_hidden_predictor, dropout=args.dropout)
        self.apply(weights_init)

    def forward(self, x):
        pred = self.predictor((x, x))
        pred = torch.cat([pred[0][:, None, :], pred[1][:, None, :]], 1)
        return torch.tanh(pred)


# ======================================================================================================================

if __name__ == '__main__':
    from easydict import EasyDict
    from dataset import CircuitDataset
    from utils import *
    from utils import plot_circuit
    import matplotlib.pyplot as plt

    args = EasyDict()
    args.len_hidden = 400
    args.len_hidden_predictor = 512
    args.len_node_attr = 11
    args.len_edge_attr = 20
    args.gnn_layers = 3
    args.use_gpu = True
    args.dropout = 0.0
    model = CircuitGNN(args)
    model.eval()

    data_loader = DataLoader(
        dataset=CircuitDataset(data_root='./data', num_block=3),
        batch_size=64,
        shuffle=True,
    )

    for data, label, raw in data_loader:
        node, edge, adj = data
        pred = model(data)
        print(pred.shape)
        loss = F.l1_loss(pred, label)
        print('loss', loss.item())

        for p, l, r in zip(to_np(pred), to_np(label), to_np(raw)):
            f, ax = plt.subplots(1, 2)
            plot_circuit(ax=ax[0], para=r)
            plot_label(ax[1], l)
            plot_pred(ax[1], p)
            plt.show()
