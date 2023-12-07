import datetime
from typing import Optional

import torch
from sklearn.decomposition import PCA


class Config:
    def __init__(self, method, hidden_size, max_epoch, prediction_days, analyze_days, name, id_train, date):
        self.method: str = method
        self.hidden_size: int = hidden_size
        self.max_epoch: int = max_epoch
        self.prediction_days: int = prediction_days
        self.analyze_days: int = analyze_days
        self.name: Optional[str] = name
        self.id_train: Optional[int] = id_train
        self.date: Optional[datetime.date] = None
        if date:
            self.date = datetime.datetime.strptime(date, "%Y-%m-%d").date()


def nice_print(text: str, num: int = 40, suffix: str = '*', suffix2: Optional[str] = None) -> None:
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
