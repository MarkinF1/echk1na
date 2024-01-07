import datetime
import os

import torch.cuda
from typing import Optional
from collections import namedtuple

from torch import optim, nn


database_name = "echkina"


class Args:
    instance = None

    def __init__(self, method, prediction_days, analyze_days, config, name, id_train, date):
        if Args.instance is None:
            self.method: str = method
            self.prediction_days: int = prediction_days
            self.analyze_days: int = analyze_days
            self.config: str = config
            self.name: Optional[str] = name
            self.id_train: Optional[int] = id_train
            self.date: Optional[datetime.date] = None
            if date:
                self.date = datetime.datetime.strptime(date, "%Y-%m-%d").date()
            Args.instance = self

    @classmethod
    def getInstance(cls):
        if cls.instance is None:
            print("Args не создан.")
            exit(-1)
        return cls.instance


class Config:
    __main_config = namedtuple('MainConfig',
                               ("checkpoint_save_dir", "checkpoint_string", "valid_objects_save_dir",
                                "valid_objects_string", "off_load_pickle_for_unit_direction", "unit_start",
                                "unit_last", "direction_start", "direction_last", "off_unit_direction",
                                "checkpoint_save", "max_count_checkpoint")
                               )
    __dataloader_config = namedtuple('DataloaderConfig',
                                     ("test_size", "train_size", "random_state")
                                     )
    __model_config = namedtuple('ModelConfig',
                                ("tp", "optimizer", "loss_function", "epoch", "lr", "lr_decay", "lr_step")
                                )
    __settings_config = namedtuple('SettingsConfig',
                                   ("off_all_prints", "print_time", "print_predict", "print_predict_step")
                                   )
    __wandb_config = namedtuple('WandbConfig',
                                ("turn_on", "project", "name")
                                )
    __fully_connected_model_config = namedtuple('FullyModelConfig',
                                                ("num_of_layers",)
                                                )
    __lstm_model_config = namedtuple('LSTMConfig',
                                     ("hidden_size", )
                                     )
    instance = None

    def __init__(self, config: dict):
        if Config.instance is None:
            self.main: Config.__main_config = self.__main_config(**config["main"])
            self.dataloader: Config.__dataloader_config = self.__dataloader_config(**config["dataloader"])
            self.model: Config.__model_config = self.__model_config(**config["model"])
            self.settings: Config.__settings_config = self.__settings_config(**config["settings"])
            self.wandb: Config.__wandb_config = self.__wandb_config(**config["wandb"])

            self.fully_model: Config.__fully_connected_model_config = (
                self.__fully_connected_model_config(**config["fully_model"]))

            self.lstm_model: Config.__lstm_model_config = self.__lstm_model_config(**config["lstm_model"])
            Config.instance = self

    @classmethod
    def getInstance(cls):
        if cls.instance is None:
            print("Config не создан.")
            exit(-1)
        return cls.instance


save_obj = namedtuple('SaveModel',
                      ('model', 'epoch', "loss_fun", "optimizer", "unit", "direction", "best_loss")
                      )


def get_optimizer(model_parameters, off_print: bool = False):
    config = Config.getInstance()

    # Создание оптимизатора
    nice_print(text=f"Создание оптимизатора {config.model.optimizer}", suffix='', suffix2='-',
               off=off_print)
    if not hasattr(optim, config.model.optimizer):
        print(f"Error: не нашел optimizer {config.model.optimizer} в optim.")
        exit(-1)

    optimizer = optim.__getattribute__(config.model.optimizer)(model_parameters, lr=config.model.lr)
    return optimizer


def get_loss_function(off_print: bool = False):
    config = Config.getInstance()

    # Определение функции потерь
    nice_print(text=f"Создание функции потерь {config.model.loss_function}", suffix='', suffix2='-',
               off=off_print)
    if not hasattr(nn, config.model.loss_function):
        print(f"Error: не нашел loss_function {config.model.loss_function} в torch.nn.")
        exit(-1)
    criterion = nn.__getattribute__(config.model.loss_function)().to(device())
    return criterion


def save_model(model, loss_fun, optimizer, epoch: int, unit: int, direction: int, best_loss: float) -> None:
    temp = save_obj(model=model,
                    epoch=epoch,
                    loss_fun=loss_fun,
                    optimizer=optimizer,
                    unit=unit,
                    direction=direction,
                    best_loss=best_loss
                    )
    os.makedirs(Config.getInstance().main.checkpoint_save_dir, mode=0o777, exist_ok=True)
    path = os.path.join(Config.getInstance().main.checkpoint_save_dir, Config.getInstance().main.checkpoint_string)
    torch.save(temp._asdict(),
               path.format(Args.getInstance().analyze_days, Args.getInstance().prediction_days, unit, direction, epoch))


def load_model(checkpoint_path: str): return save_obj(**torch.load(checkpoint_path))


def nice_print(text: str, num: int = 40, suffix: str = '*', suffix2: Optional[str] = None, off=False) -> None:
    if off:
        return

    if suffix2 is None:
        suffix2 = suffix
    print(suffix * num)
    print(text)
    print(suffix2 * num)


def date2str(date: datetime.date) -> str:
    return date.strftime("%Y-%m-%d")


def get_time(time):
    hours = time // 3600
    time -= hours * 3600

    minutes = time // 60
    seconds = time - minutes * 60

    curr_time = f""
    if hours:
        curr_time += f"{hours}h "
    if minutes:
        curr_time += f"{minutes}m "
    if time:
        curr_time += f"{seconds: .2f}s"

    return curr_time


def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def make_input_tensor(x):
    x = torch.tensor(x, dtype=torch.float32).to(device())
    mean = torch.mean(x)
    std = torch.std(x)
    x = (x - mean) / std
    return x
