import torch
import numpy as np
import scipy.sparse
import scipy.spatial
from typing import Union
from ml4co_kit import to_numpy, check_dim, atsp_greedy_decoder, atsp_2opt_local_search


class MatDIFFNetDecoder(object):
    def __init__(
        self,
        nodes_num: int = None,
        heatmap_delta: float = 1e-14,
        use_2opt: bool = True
    ):
        self.nodes_num = nodes_num
        self.heatmap_delta = heatmap_delta
        self.use_2opt = use_2opt
        self.heatmap = None
        self.points = None
        self.edge_index = None
        self.sparse = None
    
    def check_heatmap_dim(self):
        if self.sparse:
            if self.heatmap.ndim == 1:
                self.heatmap = np.expand_dims(self.heatmap, axis=0)
            check_dim(self.heatmap, 2)
        else:
            if self.heatmap.ndim == 2:
                self.heatmap = np.expand_dims(self.heatmap, axis=0)
            check_dim(self.heatmap, 3)
        
    def check_edge_index_dim(self):
        if self.edge_index.ndim == 2:
            self.edge_index = np.expand_dims(self.edge_index, axis=0)
        check_dim(self.edge_index, 3)
    
    def sparse_to_dense(self, per_heatmap_num):
        heatmap = list()
        for idx in range(self.heatmap.shape[0]):
            _edge_index = self.edge_index[idx // per_heatmap_num]
            _max_frm_value = np.max(_edge_index[0])
            _max_to_value = np.max(_edge_index[1])
            _max_index_value = max(_max_frm_value, _max_to_value)
            _heatmap = scipy.sparse.coo_matrix(
                arg1=(self.heatmap[idx], (_edge_index[0], _edge_index[1])),
                shape=(_max_index_value+1, _max_index_value+1)
            ).toarray()
            _heatmap = np.clip(
                a=_heatmap, 
                a_min=self.heatmap_delta,
                a_max=1-self.heatmap_delta
            )
            heatmap.append(_heatmap)
        self.heatmap = np.array(heatmap)
    
    def is_valid_tour(self, tour: np.ndarray):
        return sorted(tour[:-1]) == [i for i in range(self.nodes_num)]
    
    def decode(
        self,
        heatmap: Union[np.ndarray, torch.Tensor],
        dists: Union[np.ndarray, torch.Tensor] = None,
        edge_index: Union[np.ndarray, torch.Tensor] = None,
        sparse: bool = False,
        per_heatmap_num: int = 1,
    ) -> np.ndarray:
        # np.narray type
        self.sparse = sparse
        self.heatmap = to_numpy(heatmap)
        self.dists = to_numpy(dists)
        self.edge_index = to_numpy(edge_index)
        
        # check heatmap
        self.check_heatmap_dim()
        
        # sparse to dense
        if sparse:
            self.check_edge_index_dim()
            self.sparse_to_dense(per_heatmap_num)
        else:
            self.heatmap = np.clip(
                a=self.heatmap,
                a_min=self.heatmap_delta,
                a_max=1-self.heatmap_delta
            )

        # decoding
        tours = atsp_greedy_decoder(-self.heatmap)

        # 2opt
        if self.use_2opt:
            ls_tours = list()
            for _tour, _dist in zip(tours, self.dists):
                ls_tours.append(
                    atsp_2opt_local_search(_tour, _dist)
                )
            tours = np.array(ls_tours)
        
        # check the tours
        for tour in tours:
            if not self.is_valid_tour(tour):
                raise ValueError(f"The tour {tour} is not valid!")
            if tour[-1] != 0:
                raise ValueError(f"The tour {tour} is not valid!")
        
        # dim
        if tours.shape[0] == 1:
            tours = tours[0]
        return tours