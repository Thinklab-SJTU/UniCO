import numpy as np

def greedy_search(heatmaps, dist_mats, args):
    batch_cost = 0
    batch_zeros = 0
    batch_size, n_nodes, _ = heatmaps.shape
    for idx in range(batch_size):
        heatmap = heatmaps[idx]
        dist_mat = dist_mats[idx]
        tour = [0]
        cost = 0
        while len(tour) < n_nodes:
            i = tour[-1]
            neighbours = []
            for j in range(n_nodes):
                if j != i and j not in tour:
                    neighbours.append((j, heatmap[i, j]))
            j, prob = max(neighbours, key=lambda e: e[1])
            tour.append(j)
        tour.append(0)
        for i, j in zip(tour[:-1], tour[1:]):
            cost += dist_mat[i, j]
        if cost < 1e-3:
            batch_zeros += 1
        batch_cost += cost.item()
    return (batch_cost / batch_size), batch_zeros
