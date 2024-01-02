import os
import yaml
import wandb
import pickle
from time import time
from typing import Union
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.optim as optim

from dataloader import DataLoader
from models.LSTMModel import LSTMModel
from models.FullConnectedModel import FullyConnectedNN
from supporting import Args, nice_print, date2str, get_time, Config

args: Args
config: Config
existed_models = {"fully_model": FullyConnectedNN, "lstm_model": LSTMModel}


def set_model_input_size(input_size):
    def create_model() -> Union[None, FullyConnectedNN, LSTMModel]:
        global config, existed_models

        try:
            model_config = config.__getattribute__(config.model.tp)._asdict()
            model_config["input_size"] = input_size
            model = existed_models[config.model.tp](**model_config)
        except KeyError:
            print(f"Error: модель {config.model.tp} не найдена в словаре.")
            return None
        return model
    return create_model


def save_model(model, epoch: int, analyze: int, predict: int, unit: int, direction: int) -> None:
    os.makedirs(config.main.checkpoint_save_dir, mode=0o777, exist_ok=True)
    path = os.path.join(config.main.checkpoint_save_dir, config.main.checkpoint_string)
    torch.save(model.state_dict(), path.format(analyze, predict, unit, direction, epoch))


def train() -> None:
    global args, config

    os.makedirs(config.main.valid_objects_save_dir, mode=0o777, exist_ok=True)
    valid_save_path = os.path.join(config.main.valid_objects_save_dir, config.main.valid_objects_string)
    valid_save_path = valid_save_path.format(args.analyze_days, args.prediction_days, "{0}", "{1}")

    dataloader = DataLoader(count_predictions_days=args.prediction_days,
                            count_analyze_days=args.analyze_days,
                            count_directions=config.main.direction_last - config.main.direction_start + 1,
                            count_units=config.main.unit_last - config.main.unit_start + 1)
    dataloader.train()

    create_model = set_model_input_size(dataloader.get_len_batch())

    units_directions = [(unit, direction)
                        for unit in range(config.main.unit_start, config.main.unit_last + 1)
                        for direction in range(config.main.direction_start, config.main.direction_last + 1)
                        if [unit, direction] not in config.main.off_unit_direction]

    for unit, direction in units_directions:
        nice_print(text=f"ОБУЧЕНИЕ ДЛЯ UNIT: {unit}, DIRECTION: {direction}",
                   suffix="/\\", suffix2="\\/", num=17)

        # Определение модели, функции потерь и оптимизатора
        nice_print(text=f"Создание модели {existed_models[config.model.tp].__class__.__name__}", suffix='', suffix2='-',
                   off=config.settings.off_all_prints)
        model = create_model()
        if model is None:
            exit(-1)

        nice_print(text=f"Создание функции потерь {config.model.loss_function}", suffix='', suffix2='-',
                   off=config.settings.off_all_prints)
        if not hasattr(nn, config.model.loss_function):
            print(f"Error: не нашел loss_function {config.model.loss_function} в torch.nn.")
            exit(-1)
        criterion = nn.__getattribute__(config.model.loss_function)()

        nice_print(text=f"Создание оптимизатора {config.model.optimizer}", suffix='', suffix2='-',
                   off=config.settings.off_all_prints)
        if not hasattr(optim, config.model.optimizer):
            print(f"Error: не нашел optimizer {config.model.optimizer} в optim.")
            exit(-1)
        optimizer = optim.__getattribute__(config.model.optimizer)(model.parameters(), lr=config.model.lr)

        if [unit, direction] not in config.main.off_load_pickle_for_unit_direction:
            if os.path.exists(valid_save_path.format(unit, direction)):
                nice_print(text=f"Нашел дамп валидных объектов для unit {unit} и direction {direction}.", suffix='',
                           suffix2='-')
                print("Загружаю дамп...", end="\r")
                with open(valid_save_path.format(unit, direction), "rb") as file:
                    valid_objects = pickle.load(file)
                print("Дамп загружен.\n", end="\r")
            else:
                nice_print(text=f"Дамп валидных объектов для analyze {args.analyze_days}, "
                                f"prediction {args.prediction_days}, unit {unit} и direction {direction} не найден. "
                                f"Запускаюсь с пустым массивом валидных объектов.", suffix='', suffix2='-')
                valid_objects = []
        else:
            nice_print(text=f"Дамп валидных объектов для unit {unit} и direction {direction} запрещен к загрузке. "
                            f"Запускаюсь с пустым массивом валидных объектов.", suffix='', suffix2='-')
            valid_objects = []

        epoch_time = time()
        graphic_wandb = wandb.init(
                      name=config.wandb.name.format(unit, direction),
                      project=config.wandb.project,
                      config={"epochs": config.model.epoch, "lr": config.model.lr})
        best_loss = None

        nice_print(text="Начинаю обучение.", suffix='*', off=config.settings.off_all_prints)
        for epoch in range(config.model.epoch):
            if config.settings.print_time:
                epoch_duration = time() - epoch_time
                nice_print(text=f"Epoch: {epoch + 1}/{config.model.epoch} " +
                                (f"Время одной эпохи: {get_time(epoch_duration)}" if epoch else ""),
                           suffix="=", off=config.settings.off_all_prints)

                epoch_time = time()

            if [unit, direction] not in config.main.off_load_pickle_for_unit_direction and len(valid_objects):
                train_arr = valid_objects
            else:
                train_arr = dataloader.get_by_unit_direction(unit, direction)
                valid_objects = []

            is_valid_arr = bool(len(valid_objects))

            model.train()
            first_iter_time = time()
            sum_loss = 0
            count = 0
            for x, y, objects, target, i, num in train_arr:
                count += 1
                optimizer.zero_grad()
                output = model(x, target.param1, target.param2)

                loss = criterion(output, y)
                loss.backward()
                optimizer.step()

                if config.settings.print_predict and i % config.settings.print_predict_step == 0:
                    print(f"model.predict: {output}\ntarget:{y}\nloss: {loss.item()}")

                wandb.log({"loss": loss.item()})
                sum_loss += loss.item()

                if [unit, direction] not in config.main.off_load_pickle_for_unit_direction and not is_valid_arr:
                    valid_objects.append((x, y, objects, target, i, num))

                curr_time = time()
                for_one_iter = (curr_time - first_iter_time) / (i + 1)
                print(f"[{i}/{num} {for_one_iter: .2f} it/s, "
                      f"Осталось времени: {get_time((num - i - 1) * for_one_iter)}]", end='\r')

            if not config.main.off_load_pickle_for_unit_direction and not epoch:
                with open(valid_save_path.format(unit, direction), "wb") as file:
                    pickle.dump(valid_objects, file)

            epoch_loss = sum_loss / count
            wandb.log({"loss_epoch": epoch_loss, "epoch": epoch})

            if best_loss is None or best_loss > epoch_loss:
                nice_print(text=f"Best_loss: {best_loss} ----- Current_loss: {epoch_loss} "
                                f"Save checkpoint epoch: {epoch}",
                           suffix="-")
                best_loss = epoch_loss
                save_model(model=model, epoch=epoch, analyze=args.analyze_days,
                           predict=args.prediction_days, unit=unit, direction=direction)

        try:
            graphic_wandb.finish()
        except Exception:
            print("Не удалось нормально завершить wandb_unit_%s_direction_%s" % (unit, direction))


def test():
    global args, config

    raise NotImplementedError


def eval():
    global args, config

    raise NotImplementedError


def get_days_which_can_predict():
    global args

    dataloader = DataLoader(count_predictions_days=args.prediction_days)
    dates = list(map(date2str, dataloader.check_train_id_is_valid(args.id_train)))
    if args.date:
        if date2str(args.date) in dates:
            print(f"День {args.date} валидный для насоса id {args.id_train}.")
        else:
            print(f"День {args.date} невалидный для насоса id {args.id_train}.")
    else:
        print(*dates, sep='\n')


def main():
    global args, config

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
    parser.add_argument("-n", "--name", type=str, default=None,
                        help="Название насоса.")
    parser.add_argument("-i", "--id_train", type=int, default=None,
                        help="Id насоса.")
    parser.add_argument("-d", "--date", type=str, default=None,
                        help="Дата на которую нужно совершить предсказание.")

    args = Args(**vars(parser.parse_args()))
    with open(args.config, "r") as file:
        config = Config(yaml.load(file, yaml.SafeLoader))

    functions[args.method]()


if __name__ == "__main__":
    main()
