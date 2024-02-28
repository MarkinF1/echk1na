from typing import Optional

import torch

from logger import logger
from models.BaseModel import BaseModel
from supporting import device, isValidTensor


class FullyConnectedNN(BaseModel):
    """
    Полносвязная модель
    """
    param_count = 2

    def __init__(self, input_size: int, num_of_layers: int):
        super(FullyConnectedNN, self).__init__()
        self.num_of_layers = num_of_layers
        self.input_size = input_size + self.param_count
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

        logger.info(f"INFO: FullyConnectedNN has {len(self.linear_layers)} layers.")

    def forward(self, input_: torch.Tensor, *args) -> Optional[torch.Tensor]:
        if not isValidTensor(input_):
            return None

        if len(args) != self.param_count:
            print(f"[ERROR] {self.__class__.__name__}.forward: не совпадает количество "
                  f"дополнительных параметров в args.")
            exit(1)

        input_ = input_.to(device())
        if args:
            params = torch.tensor(args, dtype=torch.float32).to(device())
            input_ = torch.concat([params, input_])
        
        for layer in self.linear_layers[:-1]:
            input_ = torch.relu(layer(input_))
        
        input_ = self.linear_layers[-1](input_)
        return input_
