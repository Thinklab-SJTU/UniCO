import jittor as jt
import jittor.nn as nn
import numpy as np

def make_positional_encoding_cosh_recur(dim, n_node, scaler=10):
    dim_phi = jt.arange(0, dim, dtype=jt.float) / dim * 2 - 1 # T = 2
    position_phi = jt.arange(0, n_node, dtype=jt.float).unsqueeze(1) / n_node * 2
    phi = dim_phi + position_phi
    
    next_period_phi_idx = phi > 1
    
    phi[next_period_phi_idx] -= 2
    
    pe = 1 / jt.cosh(phi * scaler)
    return pe

def make_positional_encoding_cosh(dim, n_node, scaler=10):
    position_phi = - jt.arange(0, n_node, dtype=jt.float).unsqueeze(1) / n_node 
    dim_phi = jt.arange(0, dim, dtype=jt.float) / dim
    pe = 1 / jt.cosh((position_phi + dim_phi) * scaler)
    return pe

def make_positional_encoding_sin(dim, n_node):
    position_phi = jt.arange(0, n_node, dtype=jt.float).unsqueeze(1) / n_node * 2 * np.pi
    dim_phi = jt.arange(0, dim, dtype=jt.float) / dim * 2 * np.pi
    pe = jt.sin(position_phi + dim_phi)
    return pe

def make_positional_encoding_x1(dim, n_node):
    position_phi = - jt.arange(0, n_node, dtype=jt.float).unsqueeze(1) / n_node 
    dim_phi = jt.arange(0, dim, dtype=jt.float) / dim
    pe = 1 / (jt.abs(position_phi + dim_phi) + 1)
    return pe

def make_positional_encoding_zero(dim, n_node):
    return jt.zeros(n_node, dim)

def make_positional_encoding_origin(dim, n_node):
    position = jt.arange(0, n_node).unsqueeze(1)
    div_term = jt.exp((jt.arange(0, dim, 2, dtype=jt.float) *
                            -(math.log(10000.0) / dim)))
    pe = jt.concat([
        jt.sin(position.float() * div_term),
        jt.cos(position.float() * div_term)
    ], dim=1)
    return pe

def make_positional_encoding(dim, n_node):
    position = jt.arange(0, n_node, dtype=jt.float).unsqueeze(1)
    omega = np.pi * 2 * position / n_node *\
                    jt.cat([
                        jt.pow(2, jt.arange(0, dim/2 - 1, dtype=jt.float)),
                        jt.zeros((1,))])
    s = jt.sin(omega)
    c = jt.cos(omega)
    pe = jt.cat([s, c], dim=1)
    return pe


class NormLinear(nn.Module):
    def __init__(self, d_in, d_out, bias=False):
        super().__init__()

        self.linear = nn.Linear(d_in, d_out, bias=bias)
        self.norm = nn.InstanceNorm1d(d_out, affine=True, track_running_stats=False)

        nn.init.normal_(self.linear.weight)
        self.linear.weight.data /= jt.sum(self.linear.weight.data * self.linear.weight.data)
        if bias: nn.init.constant_(self.linear.bias, 0)

    def execute(self, x):
        out = self.linear(x)
        transposed = out.transpose(1, 2)
        normalized = self.norm(transposed)
        back_trans = normalized.transpose(1, 2)

        return back_trans

import math
from typing import Optional, List

class PrepareForMultiHeadAttention(nn.Module):
    """
    https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/transformers/mha.py
    <a id="PrepareMHA"></a>

    ## Prepare for multi-head attention

    This module does a linear transformation and splits the vector into given
    number of heads for multi-head attention.
    This is used to transform **key**, **query**, and **value** vectors.
    """

    def __init__(self, d_input: int, heads: int, d_k: int, bias: bool):
        super().__init__()
        # Linear layer for linear transform
        self.linear = nn.Linear(d_input, heads * d_k, bias=bias)
        # Number of heads
        self.heads = heads
        # Number of dimensions in vectors in each head
        self.d_k = d_k

    def execute(self, x: jt.Var):
        # Input has shape `[seq_len, batch_size, d_model]` or `[batch_size, d_model]`.
        # We apply the linear transformation to the last dimension and split that into
        # the heads.
        head_shape = x.shape[:-1]

        # Linear transform
        x = self.linear(x)

        # Split last dimension into heads
        x = x.view(*head_shape, self.heads, self.d_k)

        # Output has shape `[seq_len, batch_size, heads, d_k]` or `[batch_size, heads, d_model]`
        return x


class MultiHeadAttention(nn.Module):
    r"""
    https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/transformers/mha.py
    <a id="MHA"></a>

    ## Multi-Head Attention Module

    This computes scaled multi-headed attention for given `query`, `key` and `value` vectors.

    $$\mathop{Attention}(Q, K, V) = \underset{seq}{\mathop{softmax}}\Bigg(\frac{Q K^\top}{\sqrt{d_k}}\Bigg)V$$

    In simple terms, it finds keys that matches the query, and gets the values of
     those keys.

    It uses dot-product of query and key as the indicator of how matching they are.
    Before taking the $softmax$ the dot-products are scaled by $\frac{1}{\sqrt{d_k}}$.
    This is done to avoid large dot-product values causing softmax to
    give very small gradients when $d_k$ is large.

    Softmax is calculated along the axis of of the sequence (or time).
    """

    def __init__(self, d_input: int, heads: int, d_model: int, dropout_prob: float = 0.1, bias: bool = True):
        """
        * `heads` is the number of heads.
        * `d_model` is the number of features in the `query`, `key` and `value` vectors.
        """

        super().__init__()

        # Number of features per head
        self.d_k = d_model // heads
        # Number of heads
        self.heads = heads

        # These transform the `query`, `key` and `value` vectors for multi-headed attention.
        self.query = PrepareForMultiHeadAttention(d_input, heads, self.d_k, bias=bias)
        self.key = PrepareForMultiHeadAttention(d_input, heads, self.d_k, bias=bias)
        self.value = PrepareForMultiHeadAttention(d_input, heads, self.d_k, bias=True)

        # Softmax for attention along the time dimension of `key`
        self.softmax = nn.Softmax(dim=1)

        self.norm = nn.InstanceNorm1d(d_model, affine=True, track_running_stats=False)

        # Output layer
        self.output = nn.Linear(d_model, d_model)
        # Dropout
        self.dropout = nn.Dropout(dropout_prob)
        # Scaling factor before the softmax
        self.scale = 1 / math.sqrt(self.d_k)

        # We store attentions so that it can be used for logging, or other computations if needed
        self.attn = None

    def get_scores(self, query: jt.Var, key: jt.Var):
        """
        ### Calculate scores between queries and keys

        This method can be overridden for other variations like relative attention.
        """

        # Calculate $Q K^\top$ or $S_{ijbh} = \sum_d Q_{ibhd} K_{jbhd}$
        # return jt.einsum('ibhd,jbhd->ijbh', query, key)
        return jt.einsum('bihd,bjhd->bijh', query, key)

    def prepare_mask(self, mask: jt.Var, query_shape: List[int], key_shape: List[int]):
        """
        `mask` has shape `[seq_len_q, seq_len_k, batch_size]`, where first dimension is the query dimension.
        If the query dimension is equal to $1$ it will be broadcasted.
        """

        assert mask.shape[0] == 1 or mask.shape[0] == query_shape[0]
        assert mask.shape[1] == key_shape[0]
        assert mask.shape[2] == 1 or mask.shape[2] == query_shape[1]

        # Same mask applied to all heads.
        mask = mask.unsqueeze(-1)

        # resulting mask has shape `[seq_len_q, seq_len_k, batch_size, heads]`
        return mask

    def execute(self, x: jt.Var,
                mask: Optional[jt.Var] = None):
        """
        `query`, `key` and `value` are the tensors that store
        collection of *query*, *key* and *value* vectors.
        They have shape `[seq_len, batch_size, d_model]`.

        `mask` has shape `[seq_len, seq_len, batch_size]` and
        `mask[i, j, b]` indicates whether for batch `b`,
        query at position `i` has access to key-value at position `j`.
        """

        # `query`, `key` and `value`  have shape `[seq_len, batch_size, d_model]`
        # seq_len, batch_size, _ = x.shape
        batch_size, seq_len, _ = x.shape

        # if mask is not None:
        #     mask = self.prepare_mask(mask, query.shape, key.shape)

        # Prepare `query`, `key` and `value` for attention computation.
        # These will then have shape `[seq_len, batch_size, heads, d_k]`.
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        # Compute attention scores $Q K^\top$.
        # This gives a Var of shape `[seq_len, seq_len, batch_size, heads]`.
        scores = self.get_scores(query, key)

        # Scale scores $\frac{Q K^\top}{\sqrt{d_k}}$
        scores *= self.scale

        # Apply mask
        # if mask is not None:
        #     scores = scores.masked_fill(mask == 0, float('-inf'))

        # $softmax$ attention along the key sequence dimension
        # $\underset{seq}{softmax}\Bigg(\frac{Q K^\top}{\sqrt{d_k}}\Bigg)$
        attn = self.softmax(scores)

        # Save attentions if debugging
        # tracker.debug('attn', attn)

        # Apply dropout
        # attn = self.dropout(attn)

        # Multiply by values
        # $$\underset{seq}{softmax}\Bigg(\frac{Q K^\top}{\sqrt{d_k}}\Bigg)V$$
        # x = jt.einsum("ijbh,jbhd->ibhd", attn, value)
        x = jt.einsum("bijh,bjhd->bihd", attn, value)

        # Save attentions for any other calculations 
        self.attn = attn.detach()

        # Concatenate multiple heads
        # x = x.reshape(seq_len, batch_size, -1)
        x = x.reshape(batch_size, seq_len, -1)

        out = self.output(x)

        transposed = out.transpose(1, 2)
        # shape: (batch, embedding, problem)

        normalized = self.norm(transposed)
        # shape: (batch, embedding, problem)

        back_trans = normalized.transpose(1, 2)

        # Output layer
        return back_trans