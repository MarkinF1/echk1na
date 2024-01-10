import os
import yaml
import torch
from typing import Union
from argparse import ArgumentParser

from dataloader import DataLoader
from models.LSTMModel import LSTMModel
from models.FullConnectedModel import FullyConnectedNN
from models.MyModel import MyModel
from supporting import Args, date2str, Config

# Реализованные модели
existed_models = {"fully_model": FullyConnectedNN, "lstm_model": LSTMModel}


def set_model_input_size(input_size: int):
    """
    Функция возвращает функцию для создания модели
    с установленным входным размером модели
    :param input_size: размер входных данных
    :return: функция
    """
    def create_model() -> Union[None, FullyConnectedNN, LSTMModel]:
        global existed_models

        config = Config.getInstance()
        try:
            model_config = config.__getattribute__(config.model.tp)._asdict()
            model_config["input_size"] = input_size
            model = existed_models[config.model.tp](**model_config)
        except KeyError:
            print(f"Error: модель {config.model.tp} не найдена в словаре.")
            return None
        return model
    return create_model


def train() -> None:
    """
    Функция тренировки модели
    """
    args = Args.getInstance()
    config = Config.getInstance()

    os.makedirs(config.main.valid_objects_save_dir, mode=0o777, exist_ok=True)
    dataloader = DataLoader(count_predictions_days=args.prediction_days,
                            count_analyze_days=args.analyze_days,
                            count_directions=config.main.direction_last - config.main.direction_start + 1,
                            count_units=config.main.unit_last - config.main.unit_start + 1)
    dataloader.train()
    create_model = set_model_input_size(dataloader.get_len_batch())

    model = MyModel(create_model_fun=create_model)
    model.train()


def test() -> None:
    """
    Функция тестирования обученной модели
    """
    args = Args.getInstance()
    config = Config.getInstance()

    raise NotImplementedError


def eval() -> None:
    """
    Функция для предсказания на определенную дату
    """

    args = Args.getInstance()
    config = Config.getInstance()

    raise NotImplementedError


def get_days_which_can_predict() -> None:
    """
    Функция выводит все даты насоса, на которые может модель предсказать,
    или проверяет конкретную дату на возможность предсказания.
    """

    args = Args.getInstance()

    dataloader = DataLoader(count_predictions_days=args.prediction_days,
                            count_analyze_days=args.analyze_days)
    dates = list(map(date2str, dataloader.check_train_id_is_valid(args.id_train)))
    if args.date:
        """
        Если есть конкретная дата, то выводит можно ли её 
        предсказать или нет (есть ли данные для предсказания) 
        """
        if date2str(args.date) in dates:
            print(f"День {args.date} валидный для насоса id {args.id_train}.")
        else:
            print(f"День {args.date} невалидный для насоса id {args.id_train}.")
    else:
        """
        Выводит все даты, на которые можно предсказать
        """
        print(*dates, sep='\n')


def main() -> None:
    """
    Загрузка аргументов и конфига, запуск выбранной функции.
    """

    functions = {
        "train": train,
        "test": test,
        "eval": eval,
        "valid": get_days_which_can_predict
    }

    parser = ArgumentParser(description="Описание программы")

    parser.add_argument("-m", "--method", type=str, default="eval", choices=list(functions.keys()),
                        help="Способ работы программы: тренировка (train), тестирование (test), проверка конкретного "
                             "варианта (eval) и вывод валидных дат для конкретного train_id (valid).")
    parser.add_argument("-p", "--prediction_days", type=int, choices=[3, 14],
                        help="На сколько дней вперед вы хотите предсказать.")
    parser.add_argument("-a", "--analyze_days", type=int,
                        help="Сколько дней взять для анализа.")
    parser.add_argument("-c", "--config", type=str, help="Установка конкретного конфига программы.")
    parser.add_argument("-i", "--id_train", type=int, default=None,
                        help="Id насоса.")
    parser.add_argument("-d", "--date", type=str, default=None,
                        help="Дата на которую нужно совершить предсказание.")
    parser.add_argument("-f", "--file", type=str, default=None, help="Файл данных формата csv.")

    args = Args(**vars(parser.parse_args()))
    with open(args.config, "r") as file:
        Config(yaml.load(file, yaml.SafeLoader))

    torch.cuda.empty_cache()
    functions[args.method]()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
