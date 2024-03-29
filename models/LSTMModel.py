from typing import Optional

import torch
from torch import nn

from supporting import device, isValidTensor
from models.BaseModel import BaseModel


class LSTMModel(BaseModel):
    def __init__(self, input_size: int, hidden_size: int):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size, self.hidden_size).to(device())
        self.fc = nn.Linear(self.hidden_size, 1).to(device())

    def forward(self, input_: torch.Tensor, *args) -> Optional[torch.Tensor]:
        if not isValidTensor(input_):
            return None

        input_ = input_.to(device())[None, ...]
        output, _ = self.lstm(input_)
        output = self.fc(output[-1])
        return output
