import torch
from torch import nn
import torch_geometric.nn as gnn

class Net(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.n_nodes = args.n_nodes
        self.depth = args.depth
        self.units = args.units
        self.device = args.device
        self.v_lin0 = nn.Linear(2, self.units)
        self.v_lins1 = nn.ModuleList([nn.Linear(self.units, self.units) for i in range(self.depth)])
        self.v_lins2 = nn.ModuleList([nn.Linear(self.units, self.units) for i in range(self.depth)])
        self.v_lins3 = nn.ModuleList([nn.Linear(self.units, self.units) for i in range(self.depth)])
        self.v_lins4 = nn.ModuleList([nn.Linear(self.units, self.units) for i in range(self.depth)])
        self.v_bns = nn.ModuleList([gnn.BatchNorm(self.units) for i in range(self.depth)])
        self.e_lin0 = nn.Linear(1, self.units)
        self.e_lins = nn.ModuleList([nn.Linear(self.units, self.units) for i in range(self.depth)])
        self.e_bns = nn.ModuleList([gnn.BatchNorm(self.units) for i in range(self.depth)])
        self.e_lin1 = nn.Linear(self.units, 1)
        self.act = args.act
    def set_batch_size(self, batch_size):
        device = self.device
        self.batch_size = batch_size
        self.edge_index = torch.LongTensor([
            [i * self.n_nodes + u for i in range(self.batch_size) for u in range(self.n_nodes) for v in range(self.n_nodes) if u != v],
            [i * self.n_nodes + v for i in range(self.batch_size) for u in range(self.n_nodes) for v in range(self.n_nodes) if u != v],
        ]).to(device)
        self.e_mask = torch.BoolTensor(
            [[u != v for v in range(self.n_nodes)] for u in range(self.n_nodes)]
        ).expand(self.batch_size, self.n_nodes, self.n_nodes).to(device)
        self.par0 = torch.zeros((self.batch_size, self.n_nodes, self.n_nodes)).to(device)
    def forward(self, x, adj):
        bs, n, _ = adj.shape
        self.n_nodes = n
        self.set_batch_size(bs)
        x = x.view(self.batch_size * self.n_nodes, 2)
        w = adj.masked_select(self.e_mask).unsqueeze(dim = -1)
        x = self.v_lin0(x)
        x = self.act(x)
        w = self.e_lin0(w)
        w = self.act(w)
        for i in range(self.depth):
            x0 = x
            x1 = self.v_lins1[i](x0)
            x2 = self.v_lins2[i](x0)
            x3 = self.v_lins3[i](x0)
            x4 = self.v_lins4[i](x0)
            w0 = w
            w1 = self.e_lins[i](w0)
            w2 = torch.sigmoid(w0)
            x = x0 + self.act(self.v_bns[i](x1 + gnn.global_max_pool(w2 * x2[self.edge_index[1]], self.edge_index[0])))
            w = w0 + self.act(self.e_bns[i](w1 + x3[self.edge_index[0]] + x4[self.edge_index[1]]))
        w = self.e_lin1(w)
        par0 = self.par0.masked_scatter(self.e_mask, w.view(-1))
        return par0
