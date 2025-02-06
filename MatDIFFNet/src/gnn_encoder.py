import math
import functools
import torch
import torch.nn.functional as F
import torch.utils.checkpoint as activation_checkpoint
from torch import nn, Tensor
from torch_sparse import SparseTensor
from .gnn_layer import GNNLayer
from .gnn_embedders import ScalarEmbeddingSine1D, ScalarEmbeddingSine3D, PositionEmbeddingSine


class GNNEncoder(nn.Module):
    """Configurable GNN Encoder
    """
    def __init__(
        self, 
        num_layers: int = 12, 
        hidden_dim: int = 256, 
        output_channels: int = 2, 
        aggregation: str = "sum", 
        norm: str = "layer",
        learn_norm: bool = True, 
        track_norm: bool = False, 
        gated: bool = True,
        sparse: bool = False, 
        use_activation_checkpoint: bool = False, 
        node_feature_only: bool = False, 
        time_embed_flag: bool = True,
    ):
        super(GNNEncoder, self).__init__()
        self.node_feature_only = node_feature_only
        self.sparse = sparse
        self.hidden_dim = hidden_dim
        self.output_channels = output_channels

        time_embed_dim = hidden_dim // 2

        self.node_embed_1 = nn.Linear(hidden_dim, hidden_dim)
        self.node_embed_2 = nn.Linear(hidden_dim, hidden_dim)
        self.edge_embed = nn.Linear(hidden_dim, hidden_dim)
        self.time_embed_flag = time_embed_flag

        mix1_init = (1 / 2) ** (1 / 2)
        mix2_init = (1 / 16) ** (1 / 2)
        mix1_weight = torch.torch.distributions.Uniform(low=-mix1_init, high=mix1_init).sample((2, hidden_dim))
        mix1_bias = torch.torch.distributions.Uniform(low=-mix1_init, high=mix1_init).sample((hidden_dim,))
        self.mix1_weight = nn.Parameter(mix1_weight)
        self.mix1_bias = nn.Parameter(mix1_bias)
        
        mix2_weight = torch.torch.distributions.Uniform(low=-mix2_init, high=mix2_init).sample((hidden_dim, 1))
        mix2_bias = torch.torch.distributions.Uniform(low=-mix2_init, high=mix2_init).sample((1,))
        self.mix2_weight = nn.Parameter(mix2_weight)
        self.mix2_bias = nn.Parameter(mix2_bias)
        
        if not node_feature_only:
            self.pos_embed = PositionEmbeddingSine(hidden_dim, normalize=True)
            self.edge_pos_embed = ScalarEmbeddingSine3D(hidden_dim, normalize=False)
        else:
            self.pos_embed = ScalarEmbeddingSine1D(hidden_dim, normalize=False)

        if self.time_embed_flag:
            self.time_embed = nn.Sequential(
                nn.Linear(hidden_dim, time_embed_dim),
                nn.ReLU(),
                nn.Linear(time_embed_dim, time_embed_dim),
            )
            self.time_embed_layers = nn.ModuleList([
                nn.Sequential(
                    nn.ReLU(),
                    nn.Linear(
                        time_embed_dim,
                        hidden_dim,
                    ),
                ) for _ in range(num_layers)
            ])
            
        self.out = nn.Sequential(
            normalization(hidden_dim),
            nn.ReLU(),
            # zero_module(
                nn.Conv2d(hidden_dim, output_channels, kernel_size=1, bias=True)
            # ),
        )
        self.layers = nn.ModuleList([
            GNNLayer(hidden_dim, aggregation, norm, learn_norm, track_norm, gated)
            for _ in range(num_layers)
        ])

        self.per_layer_out = nn.ModuleList([
            nn.Sequential(
            nn.LayerNorm(hidden_dim, elementwise_affine=learn_norm),
            nn.SiLU(),
            zero_module(
                nn.Linear(hidden_dim, hidden_dim)
            ),
            ) for _ in range(num_layers)
        ])
        self.use_activation_checkpoint = use_activation_checkpoint
        
    def dense_forward(
        self, 
        x1: Tensor, 
        x2: Tensor, 
        dists: Tensor, 
        noised_matrix: Tensor, 
        edge_index: Tensor, 
        timesteps: Tensor = None
    ) -> Tensor:
        """
        Args:
            x1: Input Source Nodes  (B x V)
            x2: Input Target Nodes  (B x V)
            dists: Input Distance Matrix  (B x V x V)
            noised_matrix: Noised Solution Matrix (B x V x V)
            edge_index: Edge indices (2 x E)
            timesteps: Input node timesteps (B)
        Returns:
            Updated edge features (B x 2 x V x V)
        """
        del edge_index
        
        # Nodes Embedding
        x1 = x1.unsqueeze(-1) # (B x V x 1)
        x2 = x2.unsqueeze(-1) # (B x V x 1)
        x = torch.cat([x1, x2], dim=-1) # (B, V, 2)
        x1, x2 = self.pos_embed(x) # (B x V x H) & (B x V x H) 
        x1, x2 = self.node_embed_1(x1), self.node_embed_2(x2) # (B x V x H) & (B x V x H) 
        
        # Edge Embedding
        two_scores = torch.stack((noised_matrix, dists), dim=3) # (B x V x V X 2)
        ms1 = torch.matmul(two_scores, self.mix1_weight) # (B x V x V X H)
        ms1 = ms1 + self.mix1_bias[None, None, None, :] # (B x V x V X H)
        ms1_activated = F.relu(ms1) # (B x V x V X H)
        ms2 = torch.matmul(ms1_activated, self.mix2_weight) # (B x V x V X 1)
        ms2 = ms2 + self.mix2_bias[None, None, None, :] # (B x V x V X 1)
        mixed_graph = ms2.squeeze(-1) # (B x V x V)
        e = self.edge_embed(self.edge_pos_embed(mixed_graph)) # (B x V x V x H)

        # Time Embedding
        if self.time_embed_flag:
            time_emb = timestep_embedding(timesteps, self.hidden_dim) # (B, H)
            time_emb = self.time_embed(time_emb) # (B, H)
        
        # Dense Graph - All Ones Matrix
        graph = torch.ones_like(noised_matrix).long()
        
        # GNN Layer
        for layer, time_layer, out_layer in zip(self.layers, self.time_embed_layers, self.per_layer_out):
            x1_in, x2_in, e_in = x1, x2, e
            x1, x2, e = layer(x1, x2, e, graph, mode="direct")
            e = e + time_layer(time_emb)[:, None, None, :]
            x1 = x1_in + x1
            x2 = x2_in + x2
            e = e_in + out_layer(e)
        
        e = self.out(e.permute((0, 3, 1, 2)))
        return e

    def forward(
        self, 
        x1: Tensor,
        x2: Tensor,
        dists: Tensor,
        graph: Tensor = None,
        edge_index: Tensor = None, 
        timesteps: Tensor = None
    ):
        if self.node_feature_only:
            raise NotImplementedError()
        else:
            if self.sparse:
                raise NotImplementedError()
            else:
                return self.dense_forward(x1, x2, dists, graph, edge_index, timesteps)
            
            
def run_sparse_layer(
    layer: nn.Module,
    time_layer: nn.Module,
    out_layer: nn.Module, 
    adj_matrix: Tensor, 
    edge_index: Tensor, 
    add_time_on_edge: bool = True
):
    def custom_forward(*inputs):
        x_in = inputs[0]
        e_in = inputs[1]
        time_emb = inputs[2]
        x, e = layer(
            x_in, e_in, adj_matrix, mode="direct", 
            edge_index=edge_index, sparse=True
        )
        if not (time_layer is None):
            if add_time_on_edge:
                e = e + time_layer(time_emb)
            else:
                x = x + time_layer(time_emb)
        x = x_in + x
        e = e_in + out_layer(e)
        return x, e
    return custom_forward


def timestep_embedding(timesteps: Tensor, dim: int, max_period: int = 10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

    
class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with torch.enable_grad():
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = torch.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads


def checkpoint(func, inputs, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.

    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)
     

class GroupNorm32(nn.GroupNorm):
    def forward(self, x: Tensor):
        return super().forward(x.float()).type(x.dtype)

  
def normalization(channels: int):
    """
    Make a standard normalization layer.

    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNorm32(32, channels)


def zero_module(module: nn.Module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module