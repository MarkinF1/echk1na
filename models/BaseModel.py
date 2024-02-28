from typing import Optional

import torch
from torch import nn


class BaseModel(nn.Module):
    param_count = 0

    def __init__(self, *args, **kwargs):
        super(BaseModel, self).__init__(*args, **kwargs)

    def forward(self, input_: torch.Tensor, *args) -> Optional[torch.Tensor]:
        ...
