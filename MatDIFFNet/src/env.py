import torch
import numpy as np
import torch.utils.data
from torch import Tensor
from typing import Union
from sortedcollections import OrderedSet
from scipy.spatial.distance import cdist
from torch.utils.data import DataLoader as GraphDataLoader
from ml4co_kit import BaseEnv, to_numpy, to_tensor, check_dim
from ml4co_kit import ATSPSolver, ATSPDataGenerator, TSPSolver


class FakeDataset(torch.utils.data.Dataset):
    def __init__(self, data_size: int):
        self.data_size = data_size

    def __len__(self):
        return self.data_size
    
    def __getitem__(self, idx: int):
        return Tensor([self.data_size])


class MatDIFFNetEnv(BaseEnv):
    def __init__(
        self,
        data_type: list = ["TSP", "ATSP", "SAT", "HCP", "VC"],
        nodes_num: int = None,
        mode: str = None,
        train_data_size: int = 128000,
        train_batch_size: int = 1,
        test_batch_size: int = 1,
        val_samples: int = 1280,
        num_workers: int = 4,
        # TSP (Uniform)
        tsp_train_path: str = None,
        tsp_val_path: str = None,
        tsp_test_path: str = None,
        # ATSP (Uniform)
        atsp_train_path: str = None,
        atsp_val_path: str = None,
        atsp_test_path: str = None,
        # ATSP (SAT)
        sat_clauses_num: list = [3, 4, 5, 6, 7],
        sat_vars_num: list = [6, 5, 5, 4, 3],
        # ATSP (HCP)
        hcp_nodes_num: int = 50,
        # ATSP (VC)
        vc_e_scale: tuple = (10, 13),
        vc_k_scale: tuple = (3, 8),
        vc_n_scale: tuple = (8, 18),
        # Device
        device: str = "cpu"
    ):
        super(MatDIFFNetEnv, self).__init__(
            name="MatDIFFNetEnv",
            mode=mode,
            train_batch_size=train_batch_size,
            val_batch_size=val_samples,
            test_batch_size=test_batch_size,
            num_workers=num_workers,
            device=device
        )
        
        # Basic
        self.nodes_num = nodes_num
        self.train_data_size = train_data_size
        self.data_type = data_type
        self.val_samples = val_samples

        # ML4CO-Kit Solvers
        self.tmp_tsp_solver = TSPSolver()
        self.tmp_atsp_solver = ATSPSolver()
        
        # TSP (Uniform)
        self.tsp_train_path = tsp_train_path
        self.tsp_val_path = tsp_val_path
        self.tsp_test_path = tsp_test_path
        self.tsp_points_train = None
        self.tsp_ref_tours_train = None
        self.tsp_start_idx_train = 0
        self.tsp_points_val = None
        self.tsp_ref_tours_val = None
        
        # ATSP (Uniform)
        self.atsp_train_path = atsp_train_path
        self.atsp_val_path = atsp_val_path
        self.atsp_test_path = atsp_test_path
        self.atsp_dists_train = None
        self.atsp_ref_tours_train = None
        self.atsp_start_idx_train = 0
        self.atsp_dists_val = None
        self.atsp_ref_tours_val = None
        
        # ATSP (SAT)
        self.sat_clauses_num = sat_clauses_num
        self.sat_vars_num = sat_vars_num
        assert len(self.sat_clauses_num) == len(self.sat_vars_num)
        
        # ATSP (HCP)
        self.hcp_nodes_num = hcp_nodes_num
        
        # ATSP (VC)
        self.vc_e_scale = vc_e_scale
        self.vc_k_scale = vc_k_scale
        self.vc_n_scale = vc_n_scale

        # load data
        self.load_data()
        
    def load_data(self):
        if self.mode == "train":
            # train dataset
            self.train_dataset = FakeDataset(data_size=self.train_data_size)
            
            if "TSP" in self.data_type:
                self.tmp_tsp_solver.from_txt(self.tsp_train_path, show_time=True, ref=True)
                self.tsp_points_train = self.tmp_tsp_solver.points
                self.tsp_ref_tours_train = self.tmp_tsp_solver.ref_tours
                
            if "ATSP" in self.data_type:
                self.tmp_atsp_solver.from_txt(self.atsp_train_path, show_time=True, ref=True)
                self.atsp_dists_train = self.tmp_atsp_solver.dists
                self.atsp_ref_tours_train = self.tmp_atsp_solver.ref_tours
                    
            # val dataset
            self.val_dataset = FakeDataset(data_size=self.val_samples)

            if "TSP" in self.data_type:
                self.tmp_tsp_solver.from_txt(self.tsp_val_path, show_time=True, ref=True)
                self.tsp_points_val = self.tmp_tsp_solver.points
                self.tsp_ref_tours_val = self.tmp_tsp_solver.ref_tours 

            if "ATSP" in self.data_type:
                self.tmp_atsp_solver.from_txt(self.atsp_val_path, show_time=True, ref=True)
                self.atsp_dists_val = self.tmp_atsp_solver.dists
                self.atsp_ref_tours_val = self.tmp_atsp_solver.ref_tours

        elif self.mode == "test":
            # test dataset
            self.test_dataset = FakeDataset(data_size=1280)
        else:
            # solve mode / none mode
            pass
    
    def train_dataloader(self):
        train_dataloader=GraphDataLoader(
            self.train_dataset, 
            batch_size=self.train_batch_size, 
            shuffle=True,
            num_workers=self.num_workers, 
            pin_memory=True,
            persistent_workers=True, 
            drop_last=True
        )
        return train_dataloader

    def val_dataloader(self):
        val_dataloader=GraphDataLoader(
            self.val_dataset, 
            batch_size=self.val_batch_size, 
            shuffle=False
        )
        return val_dataloader
    
    def test_dataloader(self):
        # force the test batch size to be 1
        self.test_batch_size = 1
        test_dataloader=GraphDataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size, 
        )
        return test_dataloader
    
    def _process_data(self, dists: Tensor, ref_tour: Tensor) -> Tensor:
        # check dim
        check_dim(dists, 2)
        check_dim(ref_tour, 1)
        
        # to numpy
        dists = to_numpy(dists)
        ref_tour = to_numpy(ref_tour)
        
        # adj matrix
        nodes_num = dists.shape[0]
        adj_matrix = np.zeros((nodes_num, nodes_num))
        for i in range(ref_tour.shape[0] - 1):
            adj_matrix[ref_tour[i], ref_tour[i + 1]] = 1
        
        # to tensor
        adj_matrix = to_tensor(adj_matrix)
        return adj_matrix

    def process_data(self, dists: Tensor, ref_tours: Tensor) -> Tensor:
        # check dim
        check_dim(dists, 3)
        check_dim(ref_tours, 2)
        
        # process data
        nodes_num = dists[0].shape[-1]
        adj_matrix_list = list() 
        for idx in range(dists.shape[0]):
            adj_matrix = self._process_data(dists[idx], ref_tours[idx])
            adj_matrix_list.append(adj_matrix)
        
        # torch.cat
        adj_matrix = torch.cat(adj_matrix_list, dim=0).to(self.device)
        adj_matrix = adj_matrix.reshape(-1, nodes_num, nodes_num)
        return adj_matrix
        
    def generate_data(self, batch_size: int) -> Union[Tensor, Tensor]:
        idx = np.random.randint(low=0, high=len(self.data_type), size=(1,))[0]
        data_type = self.data_type[idx]
        if data_type == "TSP":
            dists, ref_tours = self.generate_data_tsp(batch_size)
        elif data_type == "ATSP":
            dists, ref_tours = self.generate_data_atsp(batch_size)
        elif data_type == "SAT":
            dists, ref_tours = self.generate_data_sat(batch_size)
        elif data_type == "HCP":
            dists, ref_tours = self.generate_data_hcp(batch_size)
        elif data_type == "VC":
            dists, ref_tours = self.generate_data_vc(batch_size)
        dists = to_tensor(dists).to(self.device)
        ref_tours = to_tensor(ref_tours).to(self.device)
        return dists, ref_tours
    
    def generate_data_tsp(self, batch_size: int) -> Union[np.ndarray, np.ndarray]:
        idx = self.tsp_start_idx_train
        start_idx = idx * batch_size
        end_idx = start_idx + batch_size
        points: np.ndarray = self.tsp_points_train[start_idx : end_idx]
        ref_tours = self.tsp_ref_tours_train[start_idx : end_idx]
        if end_idx + batch_size < len(self.tsp_points_train):
            self.tsp_start_idx_train += 1
        else:
            self.tsp_start_idx_train = 0
        dists = np.array([cdist(points[idx], points[idx]) for idx in range(points.shape[0])])
        self.nodes_num = dists.shape[-1]
        return dists, ref_tours
    
    def generate_data_atsp(self, batch_size: int) -> Union[np.ndarray, np.ndarray]:
        idx = self.atsp_start_idx_train
        start_idx = idx * batch_size
        end_idx = start_idx + batch_size
        dists: np.ndarray = self.atsp_dists_train[start_idx : end_idx]
        ref_tours = self.atsp_ref_tours_train[start_idx : end_idx]
        if end_idx + batch_size < len(self.atsp_dists_train):
            self.atsp_start_idx_train += 1
        else:
            self.atsp_start_idx_train = 0
        self.nodes_num = dists.shape[-1]
        return dists, ref_tours
    
    def generate_data_sat(self, batch_size: int) -> Union[np.ndarray, np.ndarray]:
        rand_i = np.random.randint(0, len(self.sat_clauses_num))
        sat_vars_nums = self.sat_vars_num[rand_i]
        sat_clauses_nums = self.sat_clauses_num[rand_i]
        tmp_astp_generator = ATSPDataGenerator(
            num_threads=batch_size,
            data_type="sat",
            sat_vars_nums=sat_vars_nums,
            sat_clauses_nums=sat_clauses_nums
        )
        self.nodes_num = tmp_astp_generator.nodes_num
        return tmp_astp_generator._generate_sat()
    
    def generate_data_hcp(self, batch_size: int) -> Union[np.ndarray, np.ndarray]:
        tmp_astp_generator = ATSPDataGenerator(
            nodes_num=self.hcp_nodes_num,
            num_threads=batch_size,
            data_type="hcp",
        )
        self.nodes_num = tmp_astp_generator.nodes_num
        return tmp_astp_generator._generate_hcp()

    def generate_data_vc(self, batch_size: int) -> Union[np.ndarray, np.ndarray]:
        E = np.random.randint(self.vc_e_scale[0], self.vc_e_scale[1])
        K = np.random.randint(self.vc_k_scale[0], self.vc_k_scale[1])
        N = np.random.randint(self.vc_n_scale[0], self.vc_n_scale[1])
        dists = list()
        ref_tours = list()
        for _ in range(batch_size):
            dist, ref_tour = gen_vertex_cover(E, K, N, calc_gt=True)
            dists.append(dist)
            ref_tours.append(ref_tour)
        return np.array(dists), np.array(ref_tours)

    def get_val_data(self) -> dict:
        val_data_dict = dict()
        for k in self.data_type:
            if k == "TSP":
                points = self.tsp_points_val
                tsp_dists_val = np.array([cdist(points[idx], points[idx]) for idx in range(self.val_samples)])
                val_data_dict["TSP"] = (tsp_dists_val[:self.val_samples], self.tsp_ref_tours_val[:self.val_samples])
            if k == "ATSP":
                val_data_dict["ATSP"] = (self.atsp_dists_val[:self.val_samples], self.atsp_ref_tours_val[:self.val_samples])
            if k == "SAT":
                val_data_dict["SAT"] = self.generate_data_sat(self.val_samples)
            if k == "HCP":
                val_data_dict["HCP"] = self.generate_data_hcp(self.val_samples)
            if k == "VC":
                val_data_dict["VC"] = self.generate_data_hcp(self.val_samples)
        return val_data_dict


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