from ctypes import CDLL, c_int, c_double, byref
import numpy as np
import tsplib95
import lkh
import torch as th

class BaseSolver:
    def __init__(self, n, lib_path="../utils/libtsp.so", scaler=1e6):
        '''
        n int: number of nodes in the problems
        '''
        self.lib_tsp = CDLL(lib_path)

        self.n = n
        self.path = (c_int * n)(*(list(range(n))))
        self.cost = c_double(0)
        self.scaler = scaler

    def _make_problem(self, dist):
        return (c_double * (self.n * self.n))(*dist.reshape(self.n * self.n).tolist()) 

    def get_cost(self):
        return self.cost.value

    def get_path(self):
        return list(self.path)

    def solve_random_walk(self, dist):
        pass

    def solve_nearest_neighbor(self, dist):
        '''
        dist numpy array: the distance matrix
        '''
        self.lib_tsp.nearest_neighbor(self.n, 
                self._make_problem(dist * self.scaler), self.path, byref(self.cost))
        
    def solve_nearest_insertion(self, dist):
        self.lib_tsp.nearest_insertion(self.n, 
                self._make_problem(dist * self.scaler), self.path, byref(self.cost))
    
    def solve_farthest_insertion(self, dist):
        self.lib_tsp.farthest_insertion(self.n, 
                self._make_problem(dist * self.scaler), self.path, byref(self.cost))
    
    def solve_rand_perm(self, dist):
        self.path = np.random.permutation(self.n).tolist()

    def solve_lkh(self, dist, runs=1, max_trials=1000):
        import warnings
        from copy import deepcopy
        warnings.filterwarnings('ignore')
        def parse_tsp(X, name='unknown'):
            if isinstance(X, th.Tensor):
                X = X.cpu().numpy()
            self.n = X.shape[-1]
            scale_X = X.reshape((self.n, self.n)) * 1e6
            dim = len(scale_X)
            outstr = ''
            outstr += 'NAME: %s\n' % name #problem.name
            outstr += 'TYPE: ATSP\n'
            outstr += 'COMMENT: %s\n' % name
            outstr += 'DIMENSION: %d\n' % dim #problem.dimension
            outstr += 'EDGE_WEIGHT_TYPE: EXPLICIT\n'
            outstr += 'EDGE_WEIGHT_FORMAT: FULL_MATRIX\n' #LOWER_DIAG_ROW
            outstr += 'EDGE_WEIGHT_SECTION:\n'
            for l in scale_X:
                listToStr = ' '.join([str(elem) for elem in l])
                outstr += ' %s\n' % listToStr
            #outstr += 'EDGE_DATA_FORMAT: EDGE_LIST\n'
            #outstr += 'EDGE_DATA_SECTION:\n'
            #for edge_idx, weight in edges_dict.items():
            #    outstr += f' {edge_idx[0]+1} {edge_idx[1]+1} {weight}\n'
            #outstr += '-1\n'
            return outstr
        def calc_lkh_tour_len(tsp, solver_path = "LKH"):
            def get_edge_weight(tsp, city1, city2):
                weight = tsp.get_weight(*(city1,city2))# if city1 > city2 else tsp.get_weight(*(city2,city1))
                return weight
            def calc_lkh_tour(tsp, solver_path):
                result= lkh.solve(solver_path, problem=tsp, runs=runs, max_trials=max_trials)
                lkh_path = result[0]
                lkh_path.append(lkh_path[0])
                return lkh_path
            lkh_path = calc_lkh_tour(tsp, solver_path)
            self.path = deepcopy(lkh_path)
            tour_len = 0
            while(len(lkh_path) > 1):
                start_node = lkh_path.pop()
                next_node = lkh_path[-1]
                tour_len += get_edge_weight(tsp, next_node-1, start_node-1)
            return tour_len
        tsp = tsplib95.parse(parse_tsp(dist))
        tour_len = calc_lkh_tour_len(tsp)
        tour_len *= 1e-6

        return tour_len

        # init path for trainning
        # result= lkh.solve('LKH', problem=tsp, runs=runs, max_trials=max_trials)
        # lkh_path = result[0]
        # lkh_path.append(lkh_path[0])
        # self.path = [node_idx - 1 for node_idx in lkh_path[:-1]]

if __name__ == "__main__":
    from ATSProblemDef import load_single_problem_from_file
    import warnings
    warnings.filterwarnings('ignore')
    from tqdm import tqdm

    so = CDLL("./libtsp.so")
    # n = 52
    scaler = 1e6
    n_instances = 10
    file_dir = 'test_set/20_10000'
    total_cost = 0
    zeros = 0
    
    '''
    path = (c_int * 4)(*path)
    dist = (c_double * 16)(*dist)
    cost = c_double(5)
    
    so.nearest(4, dist, path, byref(cost))
    
    print("cost is", cost)
    
    true_cost = 0
    print("path", list(path))
    for i in range(4):
        true_cost += dist[path[i]*4 + path[(i+1)%4]]
    print("cost value by c++", cost.value)
    print("cost value by python", true_cost)
    '''

    for i in tqdm(range(n_instances)):
        filename = file_dir + f'/{i+5000}.atsp'
        dist = load_single_problem_from_file(filename, None, scaler).numpy()
        # print(dist)
        n = dist.shape[0]
        solver = BaseSolver(n, lib_path='./libtsp.so')
        length = solver.solve_lkh(dist, runs=1, max_trials=500)
        # solver.solve_farthest_insertion(dist)
        # solver.solve_nearest_neighbor(dist)
        # tour = solver.get_path()
        # print(tour)
        length = solver.get_cost() / scaler
        # length = 0
        # for i, j in zip(tour[:-1], tour[1:]):
        #     length += dist[i, j]
        total_cost += length
        # total_cost += solver.get_cost()
        if length < 1e-3:
            zeros += 1

    print(f'Avg_gt_cost: {total_cost / n_instances:.5f}')
    print(f'ZEROS: {zeros}')
