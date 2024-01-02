from torch import nn

from models.BaseModel import BaseModel


class LSTMModel(BaseModel):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size, self.hidden_size)
        self.fc = nn.Linear(self.hidden_size, output_size)

    def forward(self, input, *args):
        output, _ = self.lstm(input)
        output = self.fc(output[-1])
        return output
