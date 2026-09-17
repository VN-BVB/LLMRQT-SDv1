import torch
from torch import nn

class RMSNorm(nn.Module):
    def __init__(self, weight, eps=1e-6):
        super().__init__()
        self.weight = weight
        self.variance_epsilon = eps
    # output = (x /sqrt ( mean (x ^2) + epsilon )) * weight
    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        inv = torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = hidden_states * inv
        return self.weight * hidden_states.to(input_dtype)    


class LayerNorm(nn.Module):
    def __init__(self, weight, bias=None, eps=1e-5):
        super().__init__()
        self.weight = weight
        self.bias = bias
        self.variance_epsilon = eps

    # output = ((x - mean(x)) / sqrt(mean((x - mean(x)) ^ 2) + epsilon)) * weight + bias
    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        mean = hidden_states.mean(dim=-1, keepdim=True)
        variance = (hidden_states - mean).pow(2).mean(dim=-1, keepdim=True)
        hidden_states = (hidden_states - mean) * torch.rsqrt(
            variance + self.variance_epsilon
        )
        hidden_states = self.weight * hidden_states.to(input_dtype)
        if self.bias is not None:
            hidden_states = hidden_states + self.bias
        return hidden_states
