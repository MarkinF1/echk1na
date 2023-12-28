import datetime
import torch.cuda
from typing import Optional
from collections import namedtuple


class Args:
    def __init__(self, method, prediction_days, analyze_days, config, name, id_train, date):
        self.method: str = method
        self.prediction_days: int = prediction_days
        self.analyze_days: int = analyze_days
        self.config: str = config
        self.name: Optional[str] = name
        self.id_train: Optional[int] = id_train
        self.date: Optional[datetime.date] = None
        if date:
            self.date = datetime.datetime.strptime(date, "%Y-%m-%d").date()


class Config:
    __main_config = namedtuple('MainConfig',
                               ("checkpoint_save_dir", "checkpoint_string", "valid_objects_save_dir",
                                "valid_objects_string", "off_load_pickle_for_unit_direction", "unit_start",
                                "unit_last", "direction_start", "direction_last", "off_unit_direction")
                               )
    __dataloader_config = namedtuple('DataloaderConfig',
                                     ("test_size", "valid_size", "train_size")
                                     )
    __model_config = namedtuple('ModelConfig',
                                ("tp", "optimizer", "loss_function", "epoch", "lr")
                                )
    __settings_config = namedtuple('SettingsConfig',
                                   ("off_all_prints", "print_time", "print_predict", "print_predict_step")
                                   )
    __wandb_config = namedtuple('WandbConfig',
                                ("project", "name")
                                )
    __fully_connected_model_config = namedtuple('FullyModelConfig',
                                                ("num_of_layers",)
                                                )
    __lstm_model_config = namedtuple('LSTMConfig',
                                     ("hidden_size", )
                                     )

    def __init__(self, config: dict):
        self.main = self.__main_config(**config["main"])
        self.dataloader = self.__dataloader_config(**config["dataloader"])
        self.model = self.__model_config(**config["model"])
        self.settings = self.__settings_config(**config["settings"])
        self.wandb = self.__wandb_config(**config["wandb"])
        self.fully_model = self.__fully_connected_model_config(**config["fully_model"])
        self.lstm_model = self.__lstm_model_config(**config["lstm_model"])


def nice_print(text: str, num: int = 40, suffix: str = '*', suffix2: Optional[str] = None, off=False) -> None:
    if off:
        return

    if suffix2 is None:
        suffix2 = suffix
    print()
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
