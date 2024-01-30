from tqdm import tqdm
import numpy as np
import torch
from torch import optim
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import sys

sys.path.insert(0, "..")

from utils.generate_data import *
from utils.ATSProblemDef import get_random_problems, load_single_problem_from_file
from greedy import greedy_search
from model import Net
from utils.utils import Dict

def tsp_gen(batch_size, n_nodes):
    batch_size = batch_size
    problem_pool = ['atsp_triangle', 'tsp_euc', 'hcp', '3sat']
    problem_type_idx = np.random.randint(len(problem_pool))
    problem_type = problem_pool[problem_type_idx]
    if problem_type == "atsp_triangle":
        node_cnt = n_nodes
        problems = get_random_problems(batch_size, node_cnt).cpu().numpy()
    elif problem_type == "tsp_euc":
        node_cnt = n_nodes
        problems = [gen_Euclidean(node_cnt, 2) for _ in range(batch_size)]
    elif problem_type == "hcp":
        node_cnt = n_nodes
        problems = [gen_hcp(node_cnt, np.random.rand() * 0.2 + 0.1) for _ in range(batch_size)]
    elif problem_type == "3sat":
        if n_nodes == 20:
            num_clauses = [3, 2, 2] # n~20
            num_vars = [3, 4, 5]
        elif n_nodes == 50:
            num_clauses = [3, 4, 5, 6, 7] # n_node: [39, 55]
            num_vars = [6, 5, 5, 4, 3]
        elif n_nodes == 100:
            num_clauses = [9, 8, 7, 6, 5]
            num_vars = [5, 6, 7, 8, 9]
        else:
            raise NotImplementedError("Problem scale {} is not implemented.".format(n_nodes))
        rand_idx = np.random.randint(0, len(num_clauses))
        node_cnt = 2 * num_clauses[rand_idx] * num_vars[rand_idx] + num_clauses[rand_idx]
        problems = [gen_3sat(num_clauses[rand_idx], num_vars[rand_idx]) for _ in range(batch_size)]
    else:
        raise NotImplementedError("Problem type {} is not implemented.".format(problem_type))
    n_nodes = problems[0].shape[-1]
    x = torch.rand(batch_size, n_nodes, 2).to(device)
    adj = torch.Tensor(np.array(problems)).to(device)
    return x, adj

def tsp_sample(adj, ze, mode='softmax', samples=1, epsilon=0., device=None): # epsilon exploration
    assert mode in ['softmax', 'greedy']
    if mode == 'greedy':
        assert samples == 1
    batch_size, n_nodes, _ = adj.shape
    zex = ze.expand((samples, batch_size, n_nodes, n_nodes))
    adj_flat = adj.view(batch_size, n_nodes * n_nodes).expand((samples, batch_size, n_nodes * n_nodes))
    idx = torch.arange(n_nodes).expand((samples, batch_size, n_nodes)).to(device)
    mask = torch.ones((samples, batch_size, n_nodes), dtype = torch.bool).to(device)
    maskFalse = torch.zeros((samples, batch_size, 1), dtype = torch.bool).to(device)
    v0 = u = torch.randint(n_nodes, dtype=torch.long, size=(samples, batch_size, 1)).to(device)
    mask.scatter_(dim=-1, index=u, src=maskFalse)
    y = []
    if mode == 'softmax':
        logp, logq = [], []
    else:
        sol = [u]
    for i in range(1, n_nodes):
        zei = zex.gather(dim=-2, index=u.unsqueeze(dim=-1).expand((samples, batch_size, 1, n_nodes))) \
                                        .squeeze(dim=-2).masked_select(mask.clone()) \
                                        .view(samples, batch_size, n_nodes - i)
        if mode == 'softmax':
            pei = F.softmax(zei, dim=-1)
            qei = epsilon / (n_nodes - i) + (1. - epsilon) * pei
            vi = qei.view(samples * batch_size, n_nodes - i).multinomial(num_samples=1, replacement=True) \
                    .view(samples, batch_size, 1)
            logp.append(torch.log(pei.gather(dim=-1, index=vi) + 1e-8))
            logq.append(torch.log(qei.gather(dim=-1, index=vi) + 1e-8))
        elif mode == 'greedy':
            vi = zei.argmax(dim=-1, keepdim=True)
        v = idx.masked_select(mask).view(samples, batch_size, n_nodes - i).gather(dim=-1, index=vi)
        y.append(adj_flat.gather(dim=-1, index=u * n_nodes + v))
        u = v
        mask.scatter_(dim=-1, index=u, src=maskFalse)
        if mode == 'greedy':
            sol.append(u)
    y.append(adj_flat.gather(dim=-1, index=u * n_nodes + v0)) # ends at node v0
    y = torch.cat(y, dim = -1).sum(dim = -1).T # (batch_size, samples)
    if mode == 'softmax':
        logp = torch.cat(logp, dim=-1).sum(dim=-1).T
        logq = torch.cat(logq, dim=-1).sum(dim=-1).T
        return y, logp, logq # (batch_size, samples)
    elif mode == 'greedy':
        return y.squeeze(dim=1), torch.cat(sol, dim=-1).squeeze(dim=0) # (batch_size,)

def tsp_greedy(adj, ze, device=None):
    return tsp_sample(adj, ze, mode='greedy', device=device) # y, sol

def tsp_optim(adj, ze0, opt_fn, steps, samples, epsilon=0., device=None):
    batch_size, n_nodes, _ = adj.shape
    ze = nn.Parameter(ze0.to(device), requires_grad=True)
    opt = opt_fn([ze])
    y_means = []
    tbar = range(1, steps + 1)
    # tbar = tqdm(tbar)
    y_bl = torch.zeros((batch_size, 1)).to(device)
    for t in tbar:
        opt.zero_grad()
        y, logp, logq = tsp_sample(adj, ze, 'softmax', samples, epsilon, device=device)
        y_means.append(y.mean().item())
        # tbar.set_description(f'step={t} y_mean={y_means[-1]:.4f}')
        y_bl = y.mean(dim=-1, keepdim=True)
        J = (((y - y_bl) * torch.exp(logp - logq)).detach() * logp).mean(dim = -1).sum()
        J.backward()
        opt.step()

    # ts = np.arange(1, steps + 1)
    # sns.lineplot(x = ts, y = y_means)
    # print(y_means)
    # plt.title('E[y] vs step')
    # plt.savefig('step_score')
    return ze

def net_train(args, net):
    print(args)
    device = args.device
    best_score = 1e6
    best_epoch = 0
    net.train()
    # net.set_batch_size(args.tr_batch_size)
    opt = args.opt_outer_fn(net.parameters())
    tbar = range(1, args.tr_outer_epochs + 1)
    tbar = tqdm(tbar)
    losses = []
    for epoch in tbar:
        opt.zero_grad()
        x, adj = tsp_gen(args.tr_batch_size, args.n_nodes)
        par0 = net(x, adj)
        par1 = tsp_optim(adj, par0, args.opt_inner_fn, args.tr_inner_epochs, args.tr_inner_samples, device=device)
        par0.backward(par1.grad / args.tr_batch_size)
        opt.step()
        losses.append(tsp_greedy(adj, par1, device=device)[0].mean().item())
        tbar.set_description(f'[epoch {epoch}] score={losses[-1]:.4f}')
        if epoch % args.val_interval == 0 or epoch == 1:
            net.eval() 
            score, score_AS = net_test(args, net, mode='val')
            if score_AS < best_score:
                best_score = score_AS
                best_epoch = epoch
                print(f'New Best Epoch: {best_epoch}, Best Score: {best_score}')
                torch.save(net.state_dict(), os.path.join(args.save_dir, 'best.pt'))
            elif epoch % (args.tr_outer_epochs // 10) == 0:
                torch.save(net.state_dict(), os.path.join(args.save_dir, args.save_path.format(epoch, score, score_AS)))
            net.train()
    ts = np.arange(1, args.tr_outer_epochs + 1)
    sns.lineplot(x=ts, y=losses)
    plt.title('score vs epoch')
    plt.savefig(os.path.join(args.save_dir, 'score_epoch'))
    np.save(os.path.join(args.save_dir, 'train_scores.npy'), np.array(losses))
    return best_epoch, best_score

def net_test(args, net, mode='val'):
    if mode == 'test':
        print(args)
    device = args.device
    net.eval()
    search_func = greedy_search
    bs = getattr(args, f'{mode}_batch_size')
    inner_steps = getattr(args, f'{mode}_inner_epochs')
    inner_samples = getattr(args, f'{mode}_inner_samples')
    problems = []
    if mode == 'val':
        atsp_dir = f'../data/val_set/{args.n_nodes}_2000'
    if mode == 'test':
        atsp_dir = f'../data/test_set/{args.n_nodes}_10000'
    n_instances = len(os.listdir(atsp_dir))
    n_batches = n_instances // bs
    for i in range(n_instances):
        atsp_file = os.path.join(atsp_dir, f'{i}.atsp')
        adj = load_single_problem_from_file(atsp_file).cpu().numpy().tolist()
        problems.append(adj)

    zero_cnt_noAS = {'hcp': 0, '3sat': 0} # for decisive problems
    zero_cnt_AS = {'hcp': 0, '3sat': 0} # for decisive problems
    cls_scores_noAS = torch.zeros((4,)) # atsp, euc, hcp, 3sat
    cls_scores_AS = torch.zeros((4,)) # atsp, euc, hcp, 3sat
    overall_scores_noAS, overall_scores_AS = 0, 0
    for batch_id in tqdm(range(n_batches), desc='eval'):
        batched_problems = torch.tensor(problems[batch_id * bs : (batch_id + 1) * bs]).to(device)
        n_nodes = batched_problems.shape[-1]
        x = torch.rand(bs, n_nodes, 2).to(device)
        par0 = net(x, batched_problems)
        # par1 = par0
        batch_avg_cost_noAS, batch_zeros_noAS = search_func(par0.cpu(), batched_problems.cpu(), args) 
        par1 = tsp_optim(batched_problems, par0, args.opt_inner_fn, inner_steps, inner_samples, device=device) # This is AS
        batch_avg_cost, batch_zeros = search_func(par1.cpu(), batched_problems.cpu(), args) 

        cls_scores_AS[batch_id // (n_batches // 4)] += batch_avg_cost
        overall_scores_AS += batch_avg_cost
        if batch_id >= (n_batches // 2):
            if batch_id < (n_batches * 0.75):
                zero_cnt_AS['hcp'] += batch_zeros
            else:
                zero_cnt_AS['3sat'] += batch_zeros

        cls_scores_noAS[batch_id // (n_batches // 4)] += batch_avg_cost_noAS
        overall_scores_noAS += batch_avg_cost_noAS
        if batch_id >= (n_batches // 2):
            if batch_id < (n_batches * 0.75):
                zero_cnt_noAS['hcp'] += batch_zeros_noAS
            else:
                zero_cnt_noAS['3sat'] += batch_zeros_noAS
                    
    atsp_score_AS = cls_scores_AS[0].item() / (n_batches // 4)
    euc_score_AS = cls_scores_AS[1].item() / (n_batches // 4)
    hcp_score_AS = cls_scores_AS[2].item() / (n_batches // 4)
    sat_score_AS = cls_scores_AS[3].item() / (n_batches // 4)
    hcp_rate_AS = zero_cnt_AS['hcp'] / (n_instances // 4) * 100
    sat_rate_AS = zero_cnt_AS['3sat'] / (n_instances // 4) * 100
    overall_score_AS = overall_scores_AS / n_batches

    atsp_score_noAS = cls_scores_noAS[0].item() / (n_batches // 4)
    euc_score_noAS = cls_scores_noAS[1].item() / (n_batches // 4)
    hcp_score_noAS = cls_scores_noAS[2].item() / (n_batches // 4)
    sat_score_noAS = cls_scores_noAS[3].item() / (n_batches // 4)
    hcp_rate_noAS = zero_cnt_noAS['hcp'] / (n_instances // 4) * 100
    sat_rate_noAS = zero_cnt_noAS['3sat'] / (n_instances // 4) * 100
    overall_score_noAS = overall_scores_noAS / n_batches
    print(
        '===Raw DIMES greedy decoding===', '\n',
        "atsp_score: ", atsp_score_noAS, '\n',
        "2d_score: ", euc_score_noAS, '\n',
        "hcp_score: ", hcp_score_noAS, '\n',
        "3sat_score: ", sat_score_noAS, '\n',
        "hcp_rate: ", hcp_rate_noAS, '%\n',
        "3sat_rate: ", sat_rate_noAS, '%\n',
        'overall score: ', overall_score_noAS, '\n'
        f'===Active Search {inner_steps} steps with {inner_samples} samples===', '\n',
        "atsp_score: ", atsp_score_AS, '\n',
        "2d_score: ", euc_score_AS, '\n',
        "hcp_score: ", hcp_score_AS, '\n',
        "3sat_score: ", sat_score_AS, '\n',
        "hcp_rate: ", hcp_rate_AS, '%\n',
        "3sat_rate: ", sat_rate_AS, '%\n',
        'overall score: ', overall_score_AS
    )
    return overall_score_noAS, overall_score_AS

train_args = Dict(
    n_nodes = 20,
    opt_outer_fn = lambda par: optim.AdamW(par, lr=1e-3, weight_decay=5e-4),
    opt_inner_fn = lambda par: optim.AdamW(par, lr=0.1, weight_decay=0.),
    tr_batch_size = 100,
    tr_outer_epochs = 1000,
    tr_inner_epochs = 15,
    tr_inner_samples = 1000,
    val_interval = 5,
    val_batch_size = 100,
    val_inner_epochs = 15,
    val_inner_samples = 1000,
    act = F.leaky_relu,
    units = 64,
    depth = 8,
    save_dir = 'result/tsp{}_{}',
    save_path = 'epoch{}_{:.3f}_{:.3f}.pt',
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
)

if __name__ == '__main__':
    # Train
    day_time = datetime.datetime.now().strftime('%m%d_%H%M')
    train_args.save_dir = train_args.save_dir.format(train_args.n_nodes, day_time)
    if not os.path.exists(train_args.save_dir):
        os.mkdir(train_args.save_dir)
    device = train_args.device
    net = Net(train_args).to(device)
    best_ep, best_score = net_train(train_args, net)
    print(f'Best epoch is {best_ep}, score is {best_score}')