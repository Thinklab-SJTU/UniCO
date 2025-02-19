import torch
from torch import optim
import torch.nn.functional as F
import sys

sys.path.insert(0, "..")

from utils.generate_data import *
from model import Net
from train import net_test
from utils.utils import Dict

NUM_NODES = 20

ckpts = {
    20: '../ckpts/DIMES_20.pt',
    50: '../ckpts/DIMES_50.pt',
    100: '../ckpts/DIMES_100.pt'
}

test_args = Dict(
    n_nodes = NUM_NODES,
    opt_outer_fn = lambda par: optim.AdamW(par, lr=1e-3, weight_decay=5e-4),
    opt_inner_fn = lambda par: optim.AdamW(par, lr=0.5, weight_decay=0.),
    test_batch_size = 100,
    test_inner_epochs = 15,
    test_inner_samples = 1000,
    act = F.leaky_relu,
    units = 64,
    depth = 8,
    ckpt_path = ckpts[NUM_NODES],
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
)

if __name__ == '__main__':
    # Test
    device = test_args.device
    net = Net(test_args).to(device)
    net.load_state_dict(torch.load(test_args.ckpt_path, map_location=device))
    print(f'Model loaded: {test_args.ckpt_path}')
    net = net_test(test_args, net, mode='test')