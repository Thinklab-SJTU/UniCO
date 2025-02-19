from ctypes import CDLL, c_int, c_double, byref
import numpy as np

def solve_nearest_neighbor(heatmap, distmat):

    def _make_problem(dist):
        return (c_double * (n * n))(*dist.reshape(n * n).tolist()) 

    lib_tsp = CDLL("../utils/libtsp.so")
    n = heatmap.shape[-1]
    path = (c_int * n)(*(list(range(n))))
    cost = c_double(0)
    scaler = 1e6
    
    lib_tsp.nearest_neighbor(n, _make_problem(-heatmap * scaler), path, byref(cost))

    path = list(path)
    if path[-1] != path[0]:
        path.append(path[0])
    cost = 0
    for i, j in zip(path[:-1], path[1:]):
        cost += distmat[i, j]

    return cost, path

def greedy_search(heatmaps, dist_mats, args):
    batch_cost = 0
    batch_zeros = 0
    batch_size, n_nodes, _ = heatmaps.shape
    for idx in range(batch_size):
        heatmap = heatmaps[idx]
        dist_mat = dist_mats[idx]
        cost, path = solve_nearest_neighbor(heatmap, dist_mat)
        if cost < 1e-3:
            batch_zeros += 1
        batch_cost += cost.item()
    return (batch_cost / batch_size), batch_zeros
