import torch
from typing import List

from models.BaseModel import BaseModel
from supporting import device, nice_print


class FullyConnectedNN(BaseModel):
    """
    Полносвязная модель
    """
    param_count = 2

    def __init__(self, input_size: int, num_of_layers: int):
        super(FullyConnectedNN, self).__init__()
        self.num_of_layers = num_of_layers
        self.input_size = input_size
        self.linear_layers = torch.nn.ModuleList()
        self.__init()

    def __init(self):
        """
        Динамическое создание слоев в зависимости от
        требуемого количества.
        :return:
        """
        step = self.input_size ** (1 / (self.num_of_layers + 1))
        curr_size_input = self.input_size
        curr_size_output = int(self.input_size // step)
        sizes = [[curr_size_input, curr_size_output]]

        while curr_size_output > step:
            curr_size_input = curr_size_output
            curr_size_output = int(curr_size_output // step)
            sizes.append([curr_size_input, curr_size_output])

        sizes.append([curr_size_output, 1])

        # Уменьшение количества слоев в случае
        # превышения количества слоев
        while len(sizes) > self.num_of_layers:
            sizes[-1][0] = sizes[-2][0]
            sizes.pop(-2)

        # Увеличение количества слоев в случае
        # небольшого количества слоев
        if not len(sizes):
            print("Error: FullyConnectedNN init() len(sizes) = 0")
            exit(1)

        while len(sizes) < self.num_of_layers:
            middle = int((sizes[0][0] - sizes[0][1]) // 2)
            sizes = [[sizes[0][0], middle]] + sizes
            sizes[1][0] = middle

        for a, b in sizes:
            self.linear_layers.append(torch.nn.Linear(a, b).to(device()))

        nice_print(text=f"INFO: FullyConnectedNN has {len(self.linear_layers)} layers.", suffix='-')

#        self.hidden_size1 = 130
#        self.hidden_size2 = 60
#        self.hidden_size3 = 30
#        self.hidden_size4 = 15

#        self.fc1 = torch.nn.Linear(input_size + 2, self.hidden_size1).to(device())
#        self.fc2 = torch.nn.Linear(self.hidden_size1, self.hidden_size2).to(device())
#        self.fc3 = torch.nn.Linear(self.hidden_size2, self.hidden_size3).to(device())
#        self.fc4 = torch.nn.Linear(self.hidden_size3, self.hidden_size4).to(device())
#        self.fc5 = torch.nn.Linear(self.hidden_size4, 1).to(device())

    def forward(self, x: torch.tensor, *args):
        if args:
            params = torch.tensor(args, dtype=torch.float32)
            params = params.to(device())
            x = torch.concat([params, x])

        for layer in self.linear_layers[:-1]:
            x = torch.relu(layer(x))
        x = self.linear_layers[-1](x)
        return x
