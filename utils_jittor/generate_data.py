import numpy as np
import jittor as jt
import scipy.spatial as ssp
from sortedcollections import OrderedSet
from ml4co_kit import TSPDataGenerator, ATSPDataGenerator

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
    dist = jt.ones((num_nodes, num_nodes))
    hpath = jt.randperm(num_nodes)
    dist[hpath, hpath.roll(-1)] = 0
    num_noise_edges = int(noise_level * num_nodes * num_nodes)
    if num_noise_edges > 0:
        heads = jt.int32(np.random.choice(num_nodes, size=num_noise_edges, replace=True))
        tails = jt.int32(np.random.choice(num_nodes, size=num_noise_edges, replace=True))
        dist[heads, tails] = 0
    return dist

def gen_vertex_cover(num_edges, k, ori_N, calc_gt=False):
    '''
    num_edges: Number of edges in VC G
    k: Number of nodes as vertex cover
    ori_N: num_nodes in original VC graph G

    if self.env_params['node_cnt'] == 50:
        E = np.random.randint(10, 13) -> num_edges
        k = np.random.randint(3, 8)   -> k
        N = np.random.randint(8, 18)  -> ori_N
    elif self.env_params['node_cnt'] == 100:
        E = np.random.randint(21, 24)
        k = np.random.randint(6, 14)
        N = np.random.randint(14, 30)
    '''
    N = 4 * num_edges + k # num_nodes in the HCP graph G'
    dist = np.ones((N, N))
    vertex_cover = OrderedSet(np.random.choice(ori_N, size=k, replace=False)) # select k nodes as vertex cover
    adj_list = [OrderedSet() for _ in range(ori_N)]
    cover_ofs = 4 * num_edges
    start_idx = {} # starting index of nodes in G'
    gt_tour = []
    link_edges = []
    while True:
        valid = True
        for _ in range(num_edges):
            while True:
                if calc_gt:
                    u = vertex_cover[np.random.randint(k)]
                else:
                    u = np.random.randint(ori_N)
                v = np.random.randint(ori_N)
                if u == v:
                    continue
                if v not in adj_list[u]:
                    adj_list[u].add(v)
                    adj_list[v].add(u)
                    break
        for u in vertex_cover:
            if not adj_list[u]:
                valid = False
                break
        if valid:
            break
        else:
            adj_list = [OrderedSet() for _ in range(ori_N)]
            continue
    new_node_idx = 0
    for i in range(ori_N):
        for j in range(len(adj_list[i])):
            u, v = i, adj_list[i][j]
            start_idx[(u, v)] = new_node_idx + j * 2
        new_node_idx += (2 * len(adj_list[i]))
    for u in range(ori_N):
        for i in range(len(adj_list[u])):
            v = adj_list[u][i]
            v_idx = start_idx[(v, u)]
            u_idx = start_idx[(u, v)]
            dist[u_idx, u_idx + 1] = 0
            dist[u_idx, v_idx] = 0
            dist[v_idx, u_idx] = 0
            dist[u_idx + 1, v_idx + 1] = 0
            dist[v_idx + 1, u_idx + 1] = 0
            if i == 0:
                for j in range(k):
                    cover_node = cover_ofs + j
                    dist[cover_node, u_idx] = 0
                    link_edges.append((cover_node, u_idx))
            if i == len(adj_list[u]) - 1:
                for j in range(k):
                    cover_node = cover_ofs + j
                    dist[u_idx + 1, cover_node] = 0
                    link_edges.append((u_idx + 1, cover_node))
            else:
                dist[u_idx + 1, u_idx + 2] = 0
    cover_node = cover_ofs
    if calc_gt:
        for u in vertex_cover:
            gt_tour.append(cover_node)
            for i in range(len(adj_list[u])):                    
                v = adj_list[u][i]
                v_idx = start_idx[(v, u)]
                u_idx = start_idx[(u, v)]
                if v in vertex_cover:
                    gt_tour.extend([u_idx, u_idx + 1])
                else:
                    gt_tour.extend([u_idx, v_idx, v_idx + 1, u_idx + 1])
            cover_node += 1
        assert len(gt_tour) == N, f'{len(gt_tour)} != {N}'
        gt_tour.append(gt_tour[0])
        assert sorted(gt_tour[:-1]) == [i for i in range(N)], sorted(gt_tour[:-1])
    return dist, gt_tour

def gen_Euclidean(n, d):
    position = np.random.rand(n, d)
    dist = ssp.distance_matrix(position, position)
    return dist 


def get_iid_random_problems(node_cnt):
    ''' copied from ASTPProblemDef.py '''
    int_min = 0
    int_max = 1000 * 1000
    scaler = 1e6

    problems = jt.randint(low=int_min, high=int_max, size=(node_cnt, node_cnt))
    # shape: (batch, node, node)
    problems[jt.arange(node_cnt), jt.arange(node_cnt)] = 0

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


# generate supervised training data of TSP & ATSP for MatDIFFNet
if __name__ == "__main__":
    tsp_data_lkh = TSPDataGenerator(
        num_threads=8,
        nodes_num=50,
        data_type="uniform",
        solver="LKH",
        train_samples_num=16, # 1.28M in prctice
        val_samples_num=8,
        test_samples_num=0,
        save_path="dir/to/save"
    )

    atsp_data_lkh = ATSPDataGenerator(
        num_threads=8,
        nodes_num=50,
        data_type="uniform",
        solver="LKH",
        train_samples_num=16, # 1.28M in prctice
        val_samples_num=8,
        test_samples_num=0,
        save_path="dir/to/save"
    )
    tsp_data_lkh.generate()
    atsp_data_lkh.generate()