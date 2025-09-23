import torch
from einops import rearrange, einsum
from torch import linalg as LA

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
        x = x.to(torch.float32)                             # for numerical stability
        mean_sq = torch.mean(x**2, dim=-1, keepdim=True)    # (..., 1)
        rms = torch.sqrt(mean_sq + self.eps)                # (..., 1)
        x_normed = x / rms                                  # (..., dmodel) 
        x_scaled = x_normed * self.scale                    # (..., dmodel)
        return x_scaled.to(in_dtype)                        # back to original dtype

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
        w1x = self.linear1(x)                               # (..., dff)
        # torch.nn.Sigmoid(w1x) creates a module, not an activation. Use torch.sigmoid
        silu = w1x * torch.sigmoid(w1x)
        return self.linear2(silu * self.linear3(x))
        
class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None, dtype=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device, dtype=dtype) / d_k))
        self.register_buffer('inv_freq', inv_freq)