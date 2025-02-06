import torch
import numpy as np
from typing import List
from torch import Tensor
from ml4co_kit import (
    ATSPEvaluator, ATSPSolver, to_tensor, 
    iterative_execution, SOLVER_TYPE
)
from .model import MatDIFFNetModel


class MatDIFFNetSolver(ATSPSolver):
    def __init__(
        self, 
        model: MatDIFFNetModel, 
        seed: int = 1234,
        pretrained_path: str = None
    ):
        # basic
        super(MatDIFFNetSolver, self).__init__(solver_type=SOLVER_TYPE.ML4ATSP, scale=1)
        np.random.seed(seed=seed)
        torch.manual_seed(seed=seed)
        self.model = model
        
        # pretrain & device & mode
        if pretrained_path is not None:
            self.model.load_state_dict(torch.load(pretrained_path, map_location="cpu"))
        self.model.to(self.model.env.device).eval()
        self.model.env.mode = "solve"
        
        # solved cache
        self.solved_tours = list()
        self.dists = list()
        
    def solve(self, dists: List[np.ndarray], show_time: bool = False) -> np.ndarray:
        self.dists = dists
        self.solved_tours = list()
        for idx in iterative_execution(range, len(dists), self.solve_msg, show_time):
            self.solved_tours.append(self._solve(dists[idx]))

    def _solve(self, dist: np.ndarray) -> np.ndarray:
        # encode
        dist = to_tensor(dist).to(self.model.env.device).unsqueeze(0)
        self.model.env.nodes_num = dist.shape[-1]
        heatmap: Tensor = self.model.encode(dists=dist)
        
        # decode
        self.model.decoder.nodes_num = heatmap.shape[-1]
        return self.model.decoder.decode(heatmap=heatmap, dists=dist)
        
    def evaluate(self):
        costs = list()
        for dist, solved_tour in zip(self.dists, self.solved_tours): 
            evaluator = ATSPEvaluator(dist)
            costs.append(evaluator.evaluate(route=solved_tour))

        costs = np.array(costs)
        costs_avg = np.mean(costs)
        print(costs_avg)