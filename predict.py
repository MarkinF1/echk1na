import torch
from argparse import ArgumentParser

import yaml

from models.MyModel import MyModel
from supporting import Args, Config


def predict():
    model = MyModel()
    model.predict()


def main():
    parser = ArgumentParser(description="Описание программы")

    parser.add_argument("-m", "--method", type=str, default="eval",
                        help="Способ работы программы: тренировка (train), тестирование (test), проверка конкретного "
                             "варианта (eval) и вывод валидных дат для конкретного train_id (valid).")
    parser.add_argument("-p", "--prediction_days", type=int, choices=[3, 14],
                        help="На сколько дней вперед вы хотите предсказать.")
    parser.add_argument("-a", "--analyze_days", type=int, choices=[10, 15],
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
    predict()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
