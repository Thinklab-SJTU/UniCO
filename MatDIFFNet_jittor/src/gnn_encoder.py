import math
import jittor as jt
from jittor import nn, Var
from jittor.distributions import Uniform
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
        mix1_weight = Uniform(low=-mix1_init, high=mix1_init).sample((2, hidden_dim))
        mix1_bias = Uniform(low=-mix1_init, high=mix1_init).sample((hidden_dim,))
        self.mix1_weight = jt.Var(mix1_weight)
        self.mix1_bias = jt.Var(mix1_bias)
        
        mix2_weight = Uniform(low=-mix2_init, high=mix2_init).sample((hidden_dim, 1))
        mix2_bias = Uniform(low=-mix2_init, high=mix2_init).sample((1,))
        self.mix2_weight = jt.Var(mix2_weight)
        self.mix2_bias = jt.Var(mix2_bias)
        
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
        x1: Var, 
        x2: Var, 
        dists: Var, 
        noised_matrix: Var, 
        edge_index: Var, 
        timesteps: Var = None
    ) -> Var:
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
        x = jt.concat([x1, x2], dim=-1) # (B, V, 2)
        x1, x2 = self.pos_embed(x) # (B x V x H) & (B x V x H) 
        x1, x2 = self.node_embed_1(x1), self.node_embed_2(x2) # (B x V x H) & (B x V x H) 
        
        # Edge Embedding
        two_scores = jt.stack((noised_matrix, dists), dim=3) # (B x V x V X 2)
        ms1 = jt.matmul(two_scores, self.mix1_weight) # (B x V x V X H)
        ms1 = ms1 + self.mix1_bias[None, None, None, :] # (B x V x V X H)
        ms1_activated = nn.relu(ms1) # (B x V x V X H)
        ms2 = jt.matmul(ms1_activated, self.mix2_weight) # (B x V x V X 1)
        ms2 = ms2 + self.mix2_bias[None, None, None, :] # (B x V x V X 1)
        mixed_graph = ms2.squeeze(-1) # (B x V x V)
        e = self.edge_embed(self.edge_pos_embed(mixed_graph)) # (B x V x V x H)

        # Time Embedding
        if self.time_embed_flag:
            time_emb = timestep_embedding(timesteps, self.hidden_dim) # (B, H)
            time_emb = self.time_embed(time_emb) # (B, H)
        
        # Dense Graph - All Ones Matrix
        graph = jt.ones_like(noised_matrix).long()
        
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

    def execute(
        self, 
        x1: Var,
        x2: Var,
        dists: Var,
        graph: Var = None,
        edge_index: Var = None, 
        timesteps: Var = None
    ):
        if self.node_feature_only:
            raise NotImplementedError()
        else:
            if self.sparse:
                raise NotImplementedError()
            else:
                return self.dense_forward(x1, x2, dists, graph, edge_index, timesteps)
            

def timestep_embedding(timesteps: Var, dim: int, max_period: int = 10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Var of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Var of positional embeddings.
    """
    half = dim // 2
    freqs = jt.exp(
        -math.log(max_period) * jt.arange(start=0, end=half, dtype=jt.float32) / half
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = jt.concat([jt.cos(args), jt.sin(args)], dim=-1)
    if dim % 2:
        embedding = jt.concat([embedding, jt.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

    
class GroupNorm32(nn.GroupNorm):
    def execute(self, x: Var):
        return super().execute(x.float()).type_as(x)

  
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