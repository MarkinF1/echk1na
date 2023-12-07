import wandb
import os
from time import time

import pickle
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.optim as optim

from dataloader import DataLoader
from supporting import Config, nice_print, date2str, get_time


class LSTMModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, unit: int, direction: int):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.unit = unit
        self.direction = direction

        self.lstm = nn.LSTM(input_size, self.hidden_size)
        self.fc = nn.Linear(self.hidden_size, output_size)

    def forward(self, input):
        output, _ = self.lstm(input)
        output = self.fc(output[-1])
        return output


def set_model_sizes(input_size, hidden_size, output_size):
    def create_model(unit, direction):
        return LSTMModel(input_size, hidden_size, output_size, unit, direction)
    return create_model


def save_model(model: LSTMModel, epoch: int, analyze: int, predict: int,
               path: str = "./checkpoints/model_a%s_p%s_u%s_d%s_epoch%s.pth") -> None:
    dirs = '/'.join(path.split('/')[:-1])
    os.makedirs(dirs, mode=0o777, exist_ok=True)
    torch.save(model.state_dict(), path % (analyze, predict, model.unit, model.direction, epoch))


def train(config: Config) -> None:
    valid_save_path = "pickle_dump_valid_train_a%s_p%s.pl" % (config.analyze_days, config.prediction_days)

    dataloader = DataLoader()
    dataloader.train()

    input_size = dataloader.get_len_batch() // 2 # Размер входных данных
    hidden_size = config.hidden_size  # Размер скрытого состояния LSTM
    output_size = 1  # Размер выходных данных

    LSTM = set_model_sizes(input_size, hidden_size, output_size)
    models = {(unit, direction): LSTM(unit, direction) for unit in range(3) for direction in range(1, 4)}

    # Определение функции потерь и оптимизатора
    criterions = {(unit, direction): nn.L1Loss()
                  for unit in range(3) for direction in range(1, 4)}

    optimizers = {(unit, direction): optim.Adam(models[(unit, direction)].parameters(), lr=0.01)
                  for unit in range(3) for direction in range(1, 4)}

    best_loss = None

    if os.path.exists(valid_save_path):
        with open(valid_save_path, "rb") as file:
            valid_objects = pickle.load(file)
    else:
        valid_objects = {}

    try:
        for unit in range(3):
            for direction in range(1, 4):
                nice_print(text=f"ОБУЧЕНИЕ ДЛЯ UNIT: {unit}, DIRECTION: {direction}",
                           suffix="/\\", suffix2="\\/", num=17)
                key = (unit, direction)

                model = models[key]
                epoch_time = time()
                graphic_wandb = wandb.init(
                              name="echkina_model_unit_%s_direction_%s" % (unit, direction),
                              project="echkina",
                              config={"epochs": config.max_epoch, "lr": 0.01})

                for epoch in range(config.max_epoch):
                    epoch_duration = time() - epoch_time
                    nice_print(text=f"Epoch: {epoch + 1}/{config.max_epoch} " +
                                    (f"Время одной эпохи: {get_time(epoch_duration)}" if epoch else ""),
                               suffix="=")

                    epoch_time = time()

                    if key in valid_objects.keys():
                        train_arr = valid_objects[key]
                        is_valid_arr = True
                    else:
                        is_valid_arr = False
                        train_arr = dataloader.get_by_unit_direction(unit, direction)
                        valid_objects[key] = []

                    model.train()
                    first_iter_time = time()
                    ls = 0
                    for x, y, objects, target, i, num in train_arr:
                        optimizers[(unit, direction)].zero_grad()
                        try:
                            output = model(x)
                        except KeyError:
                            print(f"Опачки, нет такого ключа в словаре с моделями.\n"
                                  f"Unit: {unit}\nDirection: {direction}\nModels: {models}")
                            return

                        loss = criterions[(unit, direction)](output, y)
                        loss.backward()
                        optimizers[(unit, direction)].step()

                        if i % 100 == 0:
                            print(f"model.predict: {output}\ntarget:{y}\nloss: {loss.item()}")

                        wandb.log({"loss": loss.item()})
                        ls += loss.item()

                        if not is_valid_arr:
                            valid_objects[key].append((x, y, objects, target, i, num))

                        curr_time = time()
                        for_one_iter = (curr_time - first_iter_time) / (i + 1)
                        print(f"[{i}/{num} {for_one_iter: .2f} it/s, "
                              f"Осталось времени: {get_time(((num - i - 1) / for_one_iter))}]", end='\r')

                    wandb.log({"epoch": epoch, "loss_epoch": ls / num}, step=epoch)

                    if best_loss is None or best_loss > loss.item():
                        best_loss = loss.item()
                        save_model(model=model, epoch=epoch, analyze=config.analyze_days,
                                   predict=config.prediction_days)

                        nice_print(text=f"Best_loss: {best_loss} ----- Current_loss: {loss.item()} "
                                        f"Save checkpoint epoch: {epoch}",
                                   suffix="-")

                try:
                    graphic_wandb.finish()
                except Exception:
                    print("Не удалось нормально завершить wandb_unit_%s_direction_%s" % (unit, direction))

    except Exception as exp:
        print("Сохранение данных из-за прерывания: %s" % exp)

        with open(valid_save_path, "wb") as file:
            valid_objects["exception"] = exp
            pickle.dump(valid_objects, file)
        exit(1)

    with open(valid_save_path, "wb") as file:
        pickle.dump(valid_objects, file)


def test(args):
    raise NotImplementedError


def eval(args):
    raise NotImplementedError


def get_days_which_can_predict(config: Config):
    dataloader = DataLoader(count_predictions_days=config.prediction_days)
    dates = list(map(date2str, dataloader.check_train_id_is_valid(config.id_train)))
    if config.date:
        if date2str(config.date) in dates:
            print(f"День {config.date} валидный для насоса id {config.id_train}.")
        else:
            print(f"День {config.date} невалидный для насоса id {config.id_train}.")
    else:
        print(*dates, sep='\n')


def main():
    functions = {
        "train": train,
        "test": test,
        "eval": eval,
        "valid": get_days_which_can_predict
    }

    parser = ArgumentParser(description="Описание программы")

    parser.add_argument("-m", "--method", type=str, default="eval", choices=list(functions.keys()),
                        help="Способ работы программы: тренировка (train), тестирование (test), проверка конкретного "
                             "варианта (eval) и вывод валидных дат для конкретного train_id (valid)")
    parser.add_argument("-s", "--hidden_size", type=int, default=5000,
                        help="Скрытый размер нейронной сети, используется при method=train.")
    parser.add_argument("-e", "--max_epoch", type=int, default=100,
                        help="Колчиество эпох для тренировки.")
    parser.add_argument("-p", "--prediction_days", type=int, choices=[3, 14],
                        help="На сколько дней вперед вы хотите предсказать.")
    parser.add_argument("-a", "--analyze_days", type=int,
                        help="На сколько дней вперед вы хотите предсказать.")
    parser.add_argument("-n", "--name", type=str, default=None,
                        help="Название насоса.")
    parser.add_argument("-i", "--id_train", type=int, default=None,
                        help="Id насоса.")
    parser.add_argument("-d", "--date", type=str, default=None,
                        help="Дата на которую нужно совершить предсказание.")

    config = Config(**vars(parser.parse_args()))
    functions[config.method](config)


if __name__ == "__main__":
    main()
