import os
import datetime
import torch.cuda

from torch import optim, nn
from typing import Optional
from collections import namedtuple

from logger import logger

database_name = "echkina"


class Args:
    """
    Класс аргументов коммандной строки.
    Реализован типо Singleton, сначала создаем
    через (), а последующие обращения через getInstance()
    """
    instance = None

    def __init__(self, method, prediction_days, analyze_days, config, id_train, date, file):
        if Args.instance is None:
            self.method: str = method
            self.prediction_days: int = prediction_days
            self.analyze_days: int = analyze_days
            self.config: str = config
            self.id_train: Optional[int] = id_train
            self.date: Optional[datetime.date] = None
            if date:
                self.date = datetime.datetime.strptime(date, "%Y-%m-%d").date()
            self.file = file
            Args.instance = self

    @classmethod
    def getInstance(cls):
        if cls.instance is None:
            print("Args не создан.")
            exit(-1)
        return cls.instance


class Config:
    """
    Класс для содержания конфига.
    Реализован типо Singleton, сначала создаем
    через (), а последующие обращения через getInstance()
    """
    # Главные настройки
    __main_config = namedtuple('MainConfig',
                               ("checkpoint_save_dir", "checkpoint_string", "valid_objects_save_dir",
                                "valid_objects_string", "off_load_pickle_for_unit_direction", "unit_start",
                                "unit_last", "direction_start", "direction_last", "off_unit_direction",
                                "checkpoint_save", "max_count_checkpoint")
                               )
    # Настройки загрузчика
    __dataloader_config = namedtuple('DataloaderConfig',
                                     ("test_size", "train_size", "random_state")
                                     )
    # Настройки модели
    __model_config = namedtuple('ModelConfig',
                                ("tp", "optimizer", "loss_function", "epoch", "lr", "lr_decay", "lr_step")
                                )
    # Настройки работы программы
    __settings_config = namedtuple('SettingsConfig',
                                   ("off_all_prints", "print_time", "print_predict", "print_predict_step")
                                   )
    # Настройки вывода графиков на сайт wandb
    __wandb_config = namedtuple('WandbConfig',
                                ("turn_on", "project", "name")
                                )
    # Настройки полносвязной модели
    __fully_connected_model_config = namedtuple('FullyModelConfig',
                                                ("num_of_layers",)
                                                )
    # Настройки LSTM модели
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


# Класс сохранения модели
save_obj = namedtuple('SaveModel',
                      ('model', 'epoch', "loss_fun", "optimizer", "unit", "direction", "best_loss")
                      )


def get_optimizer(model_parameters, off_print: bool = False):
    """
    Создание оптимизатора назначенного из конфиг-файла.
    :param model_parameters: параметры модели (обычно через model.parameters())
    :param off_print: можно отключить вывод о том, что создается оптимизатор
    :return: объект оптимизатора
    """
    config = Config.getInstance()

    logger.debug(f"Создание оптимизатора {config.model.optimizer}.")
    if not hasattr(optim, config.model.optimizer):
        print(f"Error: не нашел optimizer {config.model.optimizer} в optim.")
        exit(-1)

    # Создание оптимизатора
    optimizer = optim.__getattribute__(config.model.optimizer)(model_parameters, lr=config.model.lr)
    return optimizer


def get_loss_function(off_print: bool = False):
    """
    Создание функции потерь назначенной из конфиг-файла.
    :param off_print: можно отключить вывод о том, что создается функция потерь
    :return: функция потерь
    """
    config = Config.getInstance()

    logger.debug(f"Создание функции потерь {config.model.loss_function}.")
    if not hasattr(nn, config.model.loss_function):
        print(f"Error: не нашел loss_function {config.model.loss_function} в torch.nn.")
        exit(-1)

    # Определение функции потерь
    criterion = nn.__getattribute__(config.model.loss_function)().to(device())
    return criterion


def save_model(model, loss_fun, optimizer, epoch: int, unit: int, direction: int, best_loss: float) -> None:
    """
    Функция сохранения модели и дополнительных параметров.
    :param model: модель
    :param loss_fun: функция потерь
    :param optimizer: оптимизатор
    :param epoch: номер эпохи
    :param unit: unit
    :param direction: direction
    :param best_loss: лучшая ошибка
    """
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


def load_model(checkpoint_path: str):
    """
    Загрузка модели.
    :param checkpoint_path: путь до чекпоинта
    :return: набор сохраненых данных
    """
    return save_obj(**torch.load(checkpoint_path))


def nice_print(text: str, num: int = 40, prefix: str = '*',
               postfix: Optional[str] = None, off=False, log_fun=None) -> None:
    """
    Красивый вывод.
    :param text: текст вывода
    :param num: количество символов до и после
    :param prefix: символ до текста
    :param postfix: символ после текста
    :param off: выключение вывода
    :param log_fun: функция для вывода текста
    """
    if off:
        return

    if postfix is None:
        postfix = prefix

    if log_fun is None:
        log_fun = print

    text = f"{prefix * num}\n{text}\n{postfix * num}"
    log_fun(text)


def date2str(date: datetime.date) -> str:
    """
    Перевод даты в строку.
    :param date: дата
    :return: строка
    """
    return date.strftime("%Y-%m-%d")


def get_time(time) -> str:
    """
    Перевод числа во время
    :param time: время в виде числа (вроде int)
    :return: строка формата "{}h {}m {}s"
    """
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


def device() -> str:
    """
    Возвращает устройство, на котором будет проходить обработка.
    :return: "cuda" или "cpu"
    """
    return "cuda" if torch.cuda.is_available() else "cpu"


def make_input_tensor(x, normalize_on: bool = True) -> torch.Tensor:
    """
    Создание тензора из массива х.
    :param x: массив данных
    :param normalize_on: нормализовывать тензор или нет
    :return: нормализованный тензор
    """
    x = torch.tensor(x, dtype=torch.float32).to(device())
    if normalize_on:
        mean = torch.mean(x)
        std = torch.std(x)
        x = (x - mean) / std

    return x
