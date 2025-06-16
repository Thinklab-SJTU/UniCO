from ctypes import CDLL, c_int, c_double, byref
import numpy as np
import tsplib95
import jittor as jt

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
