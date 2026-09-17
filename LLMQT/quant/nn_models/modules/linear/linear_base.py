import torch
class LinearBase(torch.nn.Module):
    """Base linear layer.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self, x
    ):
        raise NotImplementedError
