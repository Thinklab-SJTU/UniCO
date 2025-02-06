import torch
import torch.nn.functional as F
from typing import Tuple
from torch import nn, Tensor
from torch_sparse import SparseTensor
from torch_sparse import sum as sparse_sum
from torch_sparse import mean as sparse_mean
from torch_sparse import max as sparse_max


class GNNLayer(nn.Module):
    """Configurable GNN Layer
    Implements the Gated Graph ConvNet layer:
        h_i = ReLU ( U*h_i + Aggr.( sigma_ij, V*h_j) ),
        sigma_ij = sigmoid( A*h_i + B*h_j + C*e_ij ),
        e_ij = ReLU ( A*h_i + B*h_j + C*e_ij ),
        where Aggr. is an aggregation function: sum/mean/max.
    References:
        - X. Bresson and T. Laurent. An experimental study of neural networks for variable graphs. 
          In International Conference on Learning Representations, 2018.
        - V. P. Dwivedi, C. K. Joshi, T. Laurent, Y. Bengio, and X. Bresson. Benchmarking graph neural networks. 
          arXiv preprint arXiv:2003.00982, 2020.
    """
    
    def __init__(
        self, 
        hidden_dim: int, 
        aggregation: str = "sum", 
        norm: str = "batch", 
        learn_norm: bool = True, 
        track_norm: bool = False, 
        gated: bool = True
    ):
        """
        Args:
            hidden_dim: Hidden dimension size (int)
            aggregation: Neighborhood aggregation scheme ("sum"/"mean"/"max")
            norm: Feature normalization scheme ("layer"/"batch"/None)
            learn_norm: Whether the normalizer has learnable affine parameters (True/False)
            track_norm: Whether batch statistics are used to compute normalization mean/std (True/False)
            gated: Whether to use edge gating (True/False)
        """
        super(GNNLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.aggregation = aggregation
        self.norm = norm
        self.learn_norm = learn_norm
        self.track_norm = track_norm
        self.gated = gated
        assert self.gated, "Use gating with GCN, pass the `--gated` flag"

        self.U1 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.U2 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.V1 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.V2 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.A1 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.A2 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.C = nn.Linear(hidden_dim, hidden_dim, bias=True)
        
        self.norm_h1 = {
            "layer": nn.LayerNorm(hidden_dim, elementwise_affine=learn_norm),
            "batch": nn.BatchNorm1d(hidden_dim, affine=learn_norm, track_running_stats=track_norm)
        }.get(self.norm, None)

        self.norm_h2 = {
            "layer": nn.LayerNorm(hidden_dim, elementwise_affine=learn_norm),
            "batch": nn.BatchNorm1d(hidden_dim, affine=learn_norm, track_running_stats=track_norm)
        }.get(self.norm, None)

        self.norm_e = {
            "layer": nn.LayerNorm(hidden_dim, elementwise_affine=learn_norm),
            "batch": nn.BatchNorm1d(hidden_dim, affine=learn_norm, track_running_stats=track_norm)
        }.get(self.norm, None)
        
    def forward(
        self, 
        h1: Tensor,
        h2: Tensor, 
        e: Tensor, 
        graph: Tensor, 
        mode: str="residual", 
        edge_index: Tensor = None, 
        sparse: bool = False
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            In Dense version:
            h1: Input Source Nodes Features (B x V x H)
            h2: Input Target Nodes Features (B x V x H)
            e: Input Edge Features (B x V x V x H)
            graph: Graph Adjacency Matrices (B x V x V)
            mode: str
            In Sparse version:
            h: Input node features (V x H)
            e: Input edge features (E x H)
            graph: torch_sparse.SparseTensor
            mode: str
            edge_index: Edge indices (2 x E)
            sparse: Whether to use sparse tensors (True/False)
        Returns:
            Updated node and edge features
        """
        # torch.set_printoptions(threshold=numpy.inf)
        if not sparse:
            batch_size, num_nodes, hidden_dim = h1.shape
        else:
            raise NotImplementedError()

        h1_in = h1
        h2_in = h2
        e_in = e

        # Linear transformations for node update
        Uh1: Tensor = self.U1(h1)  # B x V x H
        Uh2: Tensor = self.U2(h2)  # B x V x H
        Vh1: Tensor = self.V1(h1)  # B x V x H
        Vh2: Tensor = self.V2(h2)  # B x V x H
        
        if not sparse:
            Wh1 = Vh1.unsqueeze(1) + Vh2.unsqueeze(2)  # B x V x V x H
            Wh2 = Vh1.unsqueeze(2) + Vh2.unsqueeze(1)  # B x V x V x H
        else:
            raise NotImplementedError()

        # Linear transformations for edge update and gating
        Ah1: Tensor = self.A1(h1)  # B x V x H, source
        Ah2: Tensor = self.A2(h2)  # B x V x H, targetR
        Ce: Tensor = self.C(e)  # B x V x V x H / E x H

        # Update edge features and compute edge gates
        if not sparse:
            e = Ah1.unsqueeze(1) + Ah2.unsqueeze(2) + Ce  # B x V x V x H
        else:
            raise NotImplementedError()
        
        # Gates
        gates = torch.sigmoid(e)  # B x V x V x H / E x H
        gates_T = torch.sigmoid(e.transpose(1, 2))
        
        # Update node features
        if not sparse:
            h1 = Uh1 + self.aggregate(Wh1, graph, gates, sparse=False)  # B x V x H
            h2 = Uh2 + self.aggregate(Wh2, graph, gates_T, sparse=False)  # B x V x H
        else:
            raise NotImplementedError()
        
        # Normalize node features
        if not sparse:
            h1 = self.norm_h1(
                h1.view(batch_size * num_nodes, hidden_dim)
            ).view(batch_size, num_nodes, hidden_dim) if self.norm_h1 else h1

            h2 = self.norm_h2(
                h2.view(batch_size * num_nodes, hidden_dim)
            ).view(batch_size, num_nodes, hidden_dim) if self.norm_h2 else h2
            
        else:
            h1 = self.norm_h1(h1) if self.norm_h1 else h1
            h2 = self.norm_h2(h2) if self.norm_h2 else h2

        # Normalize edge features
        if not sparse:
            e = self.norm_e(
                e.view(batch_size * num_nodes * num_nodes, hidden_dim)
            ).view(batch_size, num_nodes, num_nodes, hidden_dim) if self.norm_e else e
        else:
            e = self.norm_e(e) if self.norm_e else e

        # Apply non-linearity
        h1 = F.relu(h1)
        h2 = F.relu(h2)
        e = F.relu(e)

        # Make residual connection
        if mode == "residual":
            h1 = h1_in + h1
            h2 = h2_in + h2
            e = e_in + e

        return h1, h2, e
    
    def aggregate(
        self, 
        Vh: Tensor, 
        graph: Tensor, 
        gates: Tensor, 
        mode: str = None, 
        edge_index: Tensor = None, 
        sparse: bool = False
    ) -> Tensor:
        """
        Args:
            In Dense version:
            Vh: Neighborhood features (B x V x V x H)
            graph: Graph adjacency matrices (B x V x V)
            gates: Edge gates (B x V x V x H)
            mode: str
            In Sparse version:
            Vh: Neighborhood features (E x H)
            graph: torch_sparse.SparseTensor (E edges for V x V adjacency matrix)
            gates: Edge gates (E x H)
            mode: str
            edge_index: Edge indices (2 x E)
            sparse: Whether to use sparse tensors (True/False)
        Returns:
            Aggregated neighborhood features (B x V x H)
        """
        # Perform feature-wise gating mechanism
        Vh = gates * Vh  # B x V x V x H

        # Enforce graph structure through masking
        # Vh[graph.unsqueeze(-1).expand_as(Vh)] = 0

        # Aggregate neighborhood features
        if not sparse:
            if (mode or self.aggregation) == "mean":
                return torch.sum(Vh, dim=2) / (torch.sum(graph, dim=2).unsqueeze(-1).type_as(Vh))
            elif (mode or self.aggregation) == "max":
                return torch.max(Vh, dim=2)[0]
            else:
                return torch.sum(Vh, dim=2)
        else:
            sparseVh = SparseTensor(
                row=edge_index[0],
                col=edge_index[1],
                value=Vh,
                sparse_sizes=(graph.size(0), graph.size(1))
            )

        if (mode or self.aggregation) == "mean":
            return sparse_mean(sparseVh, dim=1)
        elif (mode or self.aggregation) == "max": 
            return sparse_max(sparseVh, dim=1)
        else:
            return sparse_sum(sparseVh, dim=1)