import numpy as np
import torch as th
import scipy.spatial as ssp
import os
from .base_methods import *

def count_nodes(num_clauses, num_variables):
    return 2 * num_clauses * num_variables + num_clauses

def gen_3sat(num_clauses, num_variables):
    ''' randomly generate 3sat distance matries '''
    assert num_variables>=3, "num_variables should be no smaller than 3"

    num_nodes = count_nodes(num_clauses, num_variables)
    dist = np.ones((num_nodes, num_nodes))

    # varialble node [0, 2 * num_clauses * num_variables-1]
    for v in range(num_variables):
        ofs = v * 2 * num_clauses
        for c in range(num_clauses):
            dist[ofs + 2 * c, ofs + 2 * c + 1] = 0
            dist[ofs + 2 * c + 1, ofs + 2 * c] = 0
            if c != num_clauses - 1:
                dist[ofs + 2 * c + 1, ofs + 2 * c + 2] = 0
                dist[ofs + 2 * c + 2, ofs + 2 * c + 1] = 0
        dist[ofs, (ofs + 2 * num_clauses) % (2 * num_variables * num_clauses)] = 0
        dist[ofs, (ofs + 4 * num_clauses - 1) % (2 * num_variables * num_clauses)] = 0
        dist[ofs + 2 * num_clauses - 1, (ofs + 2 * num_clauses) % (2 * num_variables * num_clauses)] = 0
        dist[ofs + 2 * num_clauses - 1, (ofs + 4 * num_clauses - 1) % (2 * num_variables * num_clauses)] = 0

    # clause node [2 * num_clauses, 2 * num_clauses + num_clauses]
    ofs_clause = 2 * num_clauses * num_variables
    for c in range(num_clauses):
        # sampling variables
        vars = np.random.choice(num_variables, size=3, replace=False)
        # sampling signs
        signs = np.random.choice(2, 3,replace=True)
        for i in range(3):
            ofs_var = vars[i] * 2 * num_clauses
            if signs[i] == 0: # x
                dist[ofs_var + 2 * c, ofs_clause + c] = 0
                dist[ofs_clause + c, ofs_var + 2 * c + 1] = 0
            else: # not x
                dist[ofs_var + 2 * c + 1, ofs_clause + c] = 0
                dist[ofs_clause + c, ofs_var + 2 * c] = 0
    return dist

def gen_hcp(num_nodes, noise_level):
    dist = th.ones((num_nodes, num_nodes))
    hpath = th.randperm(num_nodes)
    dist[hpath, hpath.roll(-1)] = 0
    num_noise_edges = int(noise_level * num_nodes * num_nodes)
    if num_noise_edges > 0:
        heads = th.LongTensor(np.random.choice(num_nodes, size=num_noise_edges, replace=True))
        tails = th.LongTensor(np.random.choice(num_nodes, size=num_noise_edges, replace=True))
        dist[heads, tails] = 0
    return dist.cpu().numpy()
    
def gen_Euclidean(n, d):
    position = np.random.rand(n, d)
    dist = ssp.distance_matrix(position, position)
    return dist 

def get_iid_random_problems(node_cnt):
    int_min = 0
    int_max = 1000 * 1000
    scaler = 1e6
    problems = th.randint(low=int_min, high=int_max, size=(node_cnt, node_cnt))
    # shape: (batch, node, node)
    problems[th.arange(node_cnt), th.arange(node_cnt)] = 0
    while True:
        old_problems = problems.clone()
        problems, _ = (problems[:, None, :] + problems[None, :, :].transpose(1,2)).min(dim=2)
        # shape: (batch, node, node)
        if (problems == old_problems).all():
            break
    # Scale
    scaled_problems = problems.float() / scaler
    return scaled_problems
    # shape: (batch, node, node)

def write_atsp(list_m, filename, dim, name="ATSP"):
    with open(filename, 'w') as f:
        #f.write( 'NAME: %s\n' % name )#problem.name
        f.write( 'TYPE: ATSP\n' )
        f.write( 'DIMENSION: %d\n' % dim )#problem.dimension
        f.write( 'EDGE_WEIGHT_TYPE: EXPLICIT\n')
        f.write( 'EDGE_WEIGHT_FORMAT: FULL_MATRIX\n' )#LOWER_DIAG_ROW
        f.write( 'EDGE_WEIGHT_SECTION:\n')
        for l in list_m:
            try:
                listToStr = '\t'.join([str(int(elem)) for elem in l])
            except:
                print(list_m)
                exit(1)
            f.write( ' %s\n' % listToStr)
        f.write('EOF\n')
    return filename


# generate dataset
def generate_tsp_file():
    from tqdm import tqdm
    scale_int = 1e6
    n = 20
    dis = "val"
    num_samples = 2000
    batch_size = 100
    num_batches = num_samples // batch_size # 20
    path_perfix = f"../data/{dis}_set/{n}_{num_samples}"
    if not os.path.exists(path_perfix):
        os.mkdir(path_perfix)

    cnt = 0

    for i in tqdm(range(num_batches)):
        if i // (num_batches // 4) == 0:
            for _ in range(batch_size):
                instance = get_iid_random_problems(n) * scale_int # atsp
                write_atsp(instance, path_perfix + f"/{cnt}.atsp", n)
                cnt += 1
        elif i // (num_batches // 4) == 1:
            for _ in range(batch_size):
                instance = gen_Euclidean(n, 2) * scale_int
                write_atsp(instance, path_perfix + f"/{cnt}.atsp", n)
                cnt += 1
        elif i // (num_batches // 4) == 2:
            for _ in range(batch_size):
                instance = gen_hcp(n, np.random.rand() * 0.2 + 0.1) * scale_int
                write_atsp(instance, path_perfix + f"/{cnt}.atsp", n)
                cnt += 1
        else:
            if n == 20:
                num_clauses = [3, 2, 2] # n~20
                num_vars = [3, 4, 5]
            elif n == 50:
                num_clauses = [3, 5, 6, 7, 4] # n~50
                num_vars = [6, 5, 4, 3, 6] 
            elif n == 100:
                num_clauses = [5, 6, 7, 8, 9] # n~10
                num_vars = [9, 8, 7, 6, 5]
            for _ in range(batch_size):
                instance = gen_3sat(num_clauses[i % len(num_clauses)], num_vars[i % len(num_clauses)]) * scale_int
                write_atsp(instance, path_perfix + f"/{cnt}.atsp", n)
                cnt += 1