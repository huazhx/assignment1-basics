import math
import torch
from einops import rearrange, einsum, pack
from torch import linalg as LA
from typing import Optional


class Linear(torch.nn.Module):
    """
    Shape of weight matrix: (out_features, in_features)
    """

    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(
            (out_features, in_features), device=device, dtype=dtype))
        self.reset_parameters()

    def reset_parameters(self):
        deviation = 2 / (self.weight.size(0) + self.weight.size(1))
        std = deviation ** 0.5
        torch.nn.init.trunc_normal_(
            self.weight, mean=0.0, std=std, a=-3*std, b=3*std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # allow x to have arbitrary leading dimensions: (..., in_features)
        return einsum(self.weight, x, 'o i, ... i -> ... o')


class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(
            (num_embeddings, embedding_dim), device=device, dtype=dtype))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=1, a=-3, b=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dtype == torch.int64, f"Input tensor must be of type int64, got {x.dtype}"
        return self.weight[x]


class RMSNorm(torch.nn.Module):
    def __init__(self, dmodel, eps=1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.scale = torch.nn.Parameter(
            torch.empty(dmodel, device=device, dtype=dtype))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.ones_(self.scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len, dmodel) or (..., dmodel)
        in_dtype = x.dtype
        # for numerical stability
        x = x.to(torch.float32)
        mean_sq = torch.mean(x**2, dim=-1, keepdim=True)    # (..., 1)
        rms = torch.sqrt(mean_sq + self.eps)                # (..., 1)
        x_normed = x / rms                                  # (..., dmodel)
        x_scaled = x_normed * self.scale                    # (..., dmodel)

        # back to original dtype
        return x_scaled.to(in_dtype)


class PositionWiseFeedForward(torch.nn.Module):
    """
    You should set dff to approximately 8/3 dmodel in your implementation, while ensuring that  the dimensionality of the inner feed-forward layer is a multiple of 64 to make good use of your hardware.
    """

    def __init__(self, dmodel, dff, device=None, dtype=None):
        super().__init__()
        self.linear1 = Linear(dmodel, dff, device=device, dtype=dtype)
        self.linear2 = Linear(dff, dmodel, device=device, dtype=dtype)
        self.linear3 = Linear(dmodel, dff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1x = self.linear1(x)                                 # (..., dff)
        # torch.nn.Sigmoid(w1x) creates a module, not an activation. Use torch.sigmoid
        silu = w1x * torch.sigmoid(w1x)
        return self.linear2(silu * self.linear3(x))


class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        """
        Args:
            theta (float): Base frequency for rotary embedding. Typical value is 10000.0.
            d_k (int): Dimension of each attention head. Must be even.
            max_seq_len (int): Maximum sequence length for which to precompute embeddings.
        """
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.precompute_cache()

    def precompute_cache(self):
        assert self.d_k % 2 == 0
        i = torch.arange(0, self.d_k // 2, dtype=torch.float32)
        theta_i = 1 / (10000.0 ** (2.0 * i / self.d_k))
        positions = torch.arange(0, self.max_seq_len, dtype=torch.float32)

        # angles = positions[:, None] * theta_i[None, :]
        # angles = rearrange(positions, 'seq_len -> seq_len ()') * rearrange(theta_i, 'dmodel -> () dmodel')
        # use outer product (or broadcasting) to get shape (max_seq_len, d_k//2)

        # (max_seq_len, d_k // 2)
        angles = torch.outer(positions, theta_i)
        cos_vals = torch.cos(angles)
        # (max_seq_len, d_k // 2)
        sin_vals = torch.sin(angles)
        # (max_seq_len, d_k // 2, 2)
        rope_cache = torch.stack((cos_vals, sin_vals), dim=-1)
        self.register_buffer('rope_cache', rope_cache)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Apply rotary positional embeddings to input tensor x.

        Args:
            x (torch.Tensor): Input tensor of shape (..., seq_len, d_k).
            token_positions (torch.Tensor): Tensor of shape (..., seq_len) indicating the position of each token.

        Returns:
            torch.Tensor: Tensor of same shape as x with rotary embeddings applied.
        """
        d_k = x.shape[-1]
        assert d_k == self.d_k, f"Input d_k {d_k} does not match layer d_k {self.d_k}"
        assert d_k % 2 == 0, "d_k must be even"
        # ensure last dim is even and use a safe reshape -> (..., seq_len, d_k//2, 2)
        orig_dtype = x.dtype
        # view_as_complex requires a real floating dtype; promote if needed
        x_real = x.to(torch.float32)
        x_pairs = rearrange(
            x_real, '... seq_len (d_k p) -> ... seq_len d_k p', p=2)

        x_complex = torch.view_as_complex(x_pairs)
        # ...use x_complex...
        # picks correct cos/sin for each token
        # (..., seq_len, d_k//2, 2)
        cos_sin = self.rope_cache[token_positions]
        cos_sin_complex = torch.view_as_complex(
            cos_sin)        # (..., seq_len, d_k//2)
        rotated_x_complex = cos_sin_complex * \
            x_complex         # (..., seq_len, d_k//2)
        rotated_x_pairs = torch.view_as_real(
            # (..., seq_len, d_k//2, 2)
            rotated_x_complex)
        rotated_x = rearrange(
            rotated_x_pairs, '... seq_len d_k p -> ... seq_len (d_k p)')
        return rotated_x.to(orig_dtype)


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Numerically stable softmax implementation that prevents overflow/underflow.
    """
    x_max = torch.max(x, dim=dim, keepdim=True).values
    x_exp = torch.exp(x - x_max)
    x_exp_sum = torch.sum(x_exp, dim=dim, keepdim=True)
    return x_exp / x_exp_sum


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Q,K,V: (batch_size, ..., seq_q, d_k), (batch_size..., seq_k, d_k), (batch_size..., seq_k, d_v)
    mask (optional): broadcastable to (..., seq_len, seq_len).
    """
    dk = Q.shape[-1]
    # compute scaled scores (..., seq_q, d_v)

    scores = Q @ K.transpose(-2, -1) / math.sqrt(dk)
    if mask is not None:
        # mask should be broadcastable to scores; True = masked
        # CAUTION: masked_fill will mask those positions where mask is True
        # then you should use ~ to invert the boolean mask
        mask_bool = mask.bool()
        scores = scores.masked_fill(~mask_bool, float("-inf"))
    # (..., seq_q, seq_k)
    attn = softmax(scores, dim=-1)
    # (..., seq_q, d_v)
    return attn @ V


class CasualMultiheadSelfAttention(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        # suppose d_v = d_k
        self.d_v = self.d_k                                     # set d_v = d_k
        self.linear_q_k_v = Linear(d_model, 2 * d_model + self.d_v)
        self.linear_out = Linear(self.d_v * num_heads, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.linear_q_k_v(x)
        q, k, v = rearrange(
            qkv, 'batch seq (head d) -> batch head seq d', head=self.num_heads, d=self.d_k,
        ).chunk(3, dim=-1)
        batch_size, n_heads, seq_len, d_k = q.shape
        # create casual mask
        mask = torch.triu(torch.ones(
            (seq_len, seq_len), device=x.device, dtype=torch.bool), diagonal=1) == 0
        attn = scaled_dot_product_attention(q, k, v, mask=mask)
        attn_merged = rearrange(
            attn, 'batch head seq d -> batch seq (head d)')
        out = self.linear_out(attn_merged)
        return out