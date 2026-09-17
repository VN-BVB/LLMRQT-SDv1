import torch
class BaseQuantizer(torch.nn.Module):
    """Base Quantzier of all Quantizer, including awq, sq, fp8.
    """

    def __init__(self):
        super().__init__()
