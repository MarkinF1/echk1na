import os
import csv
import tqdm
import torch
import wandb
import datetime
from time import time
from typing import Optional, Dict, List

from logger import logger
from dataloader import DataLoader
from db_classes import EchkinaReadyTable
from supporting import Config, nice_print, get_time, Args, get_optimizer, get_loss_function, save_obj, database_name, \
                        make_input_tensor


class MyChildModel:
    def __init__(self, model, optimizer, loss_fun, unit: int, direction: int,
                 best_loss: Optional[float] = None, epoch: int = 0, *args, **kwargs):
        self.unit = unit
        self.model = model
        self.epoch = epoch
        self.optim = optimizer
        self.loss_fun = loss_fun
        self.direction = direction
        self.best_loss = best_loss

        self.args = Args.getInstance()
        self.config = Config.getInstance()

        self.valid_validate_objects = []
        self.valid_test_objects = []
        self.save_paths: List[str] = []

    #def __str__(self):
    #    print(f"{self.__class__.__name__}(model: {self.model}, unit: {self.unit}, direction: {self.direction}, "
    #          f"epoch: {self.epoch}, best_loss: {self.best_loss})")

    def __load_valid_objects(self, path: str) -> DataLoader:
        """
        Загрузка дампа валидных объектов, если он существует.
        :param path: путь до дампа.
        """
        dataloader = DataLoader.getInstance(database=database_name)
        dataloader.train()
        if (self.config.main.off_load_pickle_for_unit_direction is None or
                [self.unit, self.direction] not in self.config.main.off_load_pickle_for_unit_direction):
            if os.path.exists(path.format(self.unit, self.direction)):
                logger.info(text=f"Нашел дамп валидных объектов для unit {self.unit} и direction {self.direction}.")
                print("Загружаю дамп...", end="\r")
                dataloader.load_pickle(self.unit, self.direction, path)
                print("Дамп загружен.\n", end="\r")
            else:
                logger.info(text=f"Дамп валидных объектов для analyze {self.args.analyze_days}, prediction "
                                f"{self.args.prediction_days}, unit {self.unit} и direction {self.direction} не найден."
                                f" Dataloader устанавливает массивы для обучения.")
                dataloader.set_unit_direction(self.unit, self.direction)
                dataloader.save_pickle()
        else:
            logger.info(text=f"Дамп валидных объектов для unit {self.unit} и direction {self.direction} запрещен к "
                            f"загрузке. Запускаюсь с пустым массивом валидных объектов.")
            dataloader.set_unit_direction(self.unit, self.direction)
            dataloader.save_pickle()
        return dataloader

    def train(self) -> None:
        valid_save_path = os.path.join(self.config.main.valid_objects_save_dir, self.config.main.valid_objects_string)
        valid_save_path = valid_save_path.format(self.config.dataloader.random_state, self.args.analyze_days,
                                                 self.args.prediction_days, "{0}", "{1}")

        nice_print(text=f"ОБУЧЕНИЕ ДЛЯ UNIT: {self.unit}, DIRECTION: {self.direction}",
                   prefix="/\\", postfix="\\/", num=17, log_fun=logger.info)

        train_arr = self.__load_valid_objects(path=valid_save_path)

        # Настройка вспомогательных параметров
        epoch_time = time()
        if self.config.wandb.turn_on:
            model_type = self.config.model.tp
            model_param_string = '_'.join(list(map(str, self.config.__getattribute__(model_type)._asdict().values())))
            graphic_wandb = wandb.init(
                          name=self.config.wandb.name.format(model_type, model_param_string,
                                                             self.config.model.optimizer, self.unit, self.direction),
                          project=self.config.wandb.project.format(self.unit, self.direction),
                          config={"epochs": self.config.model.epoch, "lr": self.config.model.lr})

        logger.info("Начинаю обучение.")
        start_epoch = self.epoch
        logger.info("Эпоха: ")
        for self.epoch in tqdm.tqdm(range(start_epoch, self.config.model.epoch)):
            self.model.train()
            train_arr.train()

            sum_loss = 0
            count = 0
            for i, (x, y, target) in enumerate(train_arr):
                count += 1
                self.optim.zero_grad()
                output = self.model(x, target.param1, target.param2)

                loss = self.loss_fun(output, y)
                loss.backward()
                self.optim.step()

                if self.config.settings.print_predict and i % self.config.settings.print_predict_step == 0:
                    logger.debug(f"model.predict: {output}\ntarget:{y}\nloss: {loss.item()}")

                sum_loss += loss.item()

            if not count:
                logger.warning(f"Не нашел валидных объектов для unit {self.unit}, direction {self.direction}. "
                      f"Продолжить [y/n]?")
                n = input()
                if n == 'y':
                    break
                else:
                    exit(-1)

            epoch_loss = sum_loss / count
            if self.config.wandb.turn_on:
                try:
                    wandb.log({"train_loss": epoch_loss, "epoch": self.epoch})
                except Exception as exp:
                    logger.error(f"Wandb: не смог сделать log. {exp}.")

            self.validate()

            if self.epoch % self.config.model.lr_step == (self.config.model.lr_step - 1):
                for g in self.optim.param_groups:
                    g["lr"] *= self.config.model.lr_decay

        if self.save_paths:
            logger.info(f"Лучшая эпоха: {self.save_paths[-1]}")

        if self.config.wandb.turn_on:
            try:
                graphic_wandb.finish()
            except Exception:
                logger.warning(f"Не удалось нормально завершить wandb_unit_{self.unit}_direction_{self.direction}.")

    def validate(self) -> None:
        self.model.eval()
        dataloader = DataLoader.getInstance(database=database_name)
        dataloader.validate()

        with torch.no_grad():
            sum_loss = 0
            count = 0
            for i, (x, y, target) in enumerate(dataloader):
                count += 1
                output = self.model(x, target.param1, target.param2)
                loss = self.loss_fun(output, y)

                if self.config.settings.print_predict and i % self.config.settings.print_predict_step == 0:
                    print(f"model.predict: {output}\ntarget:{y}\nloss: {loss.item()}")

                sum_loss += loss.item()

            if not count:
                print(f"Не нашел валидных объектов для unit {self.unit}, direction {self.direction}. "
                      f"Продолжить [y/n]?")
                n = input()
                if n == 'y':
                    return
                else:
                    exit(-1)

            epoch_loss = sum_loss / count
            if self.config.wandb.turn_on:
                try:
                    wandb.log({"valid_loss": epoch_loss})
                except Exception as exp:
                    print(f"[ERROR] Wandb: не смог сделать log. {exp}.")

            if self.best_loss is None or self.best_loss > epoch_loss:
                logger.debug(f"Best_loss: {self.best_loss} ----- Current_loss: {epoch_loss} "
                             f"Save checkpoint epoch: {self.epoch}.")
                self.best_loss = epoch_loss
                self.save()

    def test(self) -> None:
        nice_print(text=f"ТЕСТИРОВАНИЕ ДЛЯ UNIT: {self.unit}, DIRECTION: {self.direction}",
                   prefix="/\\", postfix="\\/", num=17, log_fun=logger.info)

        valid_save_path = os.path.join(self.config.main.valid_objects_save_dir, self.config.main.valid_objects_string)
        valid_save_path = valid_save_path.format(self.config.dataloader.random_state, self.args.analyze_days,
                                                 self.args.prediction_days, "{0}", "{1}")

        self.model.eval()
        dataloader = self.__load_valid_objects(valid_save_path)
        dataloader.eval()

        with torch.no_grad():
            losses = []
            count = 0
            for x, y, objects, target, i, num in tqdm.tqdm(dataloader):
                count += 1
                output = self.model(x, target.param1, target.param2)
                loss = self.loss_fun(output, y)
                loss.backward()
                losses.append(loss.item())

            epoch_loss = sum(losses) / count

        nice_print(text=f"Ошибка на тестовом наборе: {epoch_loss}.\nМаксимальная ошибка: {max(losses)}.\n"
                        f"Минимальная ошибка: {min(losses)}.", prefix='=', log_fun=logger.info)

    def predict(self, x: torch.Tensor, param1, param2) -> None:
        y = self.model(x, param1, param2)
        logger.info(text=f"Unit: {self.unit}, direction: {self.direction}, предсказание: {y}.")

    def save(self) -> None:
        temp = save_obj(model=self.model,
                        epoch=self.epoch,
                        loss_fun=self.loss_fun,
                        optimizer=self.optim,
                        unit=self.unit,
                        direction=self.direction,
                        best_loss=self.best_loss
                        )
        os.makedirs(Config.getInstance().main.checkpoint_save_dir, mode=0o777, exist_ok=True)
        path = os.path.join(Config.getInstance().main.checkpoint_save_dir, Config.getInstance().main.checkpoint_string)
        path = path.format(self.args.analyze_days, self.args.prediction_days, self.unit, self.direction, self.epoch)
        torch.save(temp._asdict(), path)
        self.save_paths.append(path)
        while len(self.save_paths) > self.config.main.max_count_checkpoint:
            os.remove(self.save_paths[0])
            self.save_paths.pop(0)


class MyModel:
    def __init__(self, create_model_fun=None):
        self.__create_model_fun = create_model_fun

        self.__model_type = None
        if self.__create_model_fun is not None:
            self.__model_type = self.__create_model_fun().__class__.__name__

        self.args = Args.getInstance()
        self.config = Config.getInstance()
        self.name = self.__class__.__name__
        self.key = "({0}, {1})"

        self.__child_models: Dict[str, MyChildModel] = {}
        self.__load()

    def train(self):
        logger.debug(f"{self.name}: train.")
        for model in self.__child_models.values():
            model.train()

    def eval(self):
        logger.debug(f"{self.name}: eval.")
        for model in self.__child_models.values():
            model.test()

    def predict(self):
        def create_obj(row):
            headers = ["id", "id_train", "id_point", "id_measure", "direction", "unit", "date", "value",
                       "alarm3", "alarm4", "param1", "param2", "arr_idx"]
            for i in range(len(row)):
                if row[i] == "":
                    row[i] = None
                elif row[i].find(".") != -1:
                    row[i] = float(row[i])
                elif row[i].split("-") == 2:
                    datetime.datetime.strptime(row[i].split(' ')[0], "%Y-%m-%d").date()
                else:
                    try:
                        row[i] = int(row[i])
                    except Exception as exp:
                        logger.exception(text=f"Не смог преобразовать {type(row[i])} в int. Exception: {exp}")
            #for i in range(6):
            #    if row[i] != "":
            #        row[i] = int(row[i])
            #    else:
            #        row[i] = None
            #if row[6] != "":
            #    row[6] = datetime.datetime.strptime(row[6].split(' ')[0], "%Y-%m-%d").date()
            #else:
            #    row[6] = None
            #for i in range(7, 12):
            #    if row[i] != "":
            #        row[i] = float(row[i])
            #    else:
            #        row[i] = None
            #if row[12] != "":
            #    row[12] = int(row[12])
            #else:
            #    row[12] = None
            row = {key: val for key, val in zip(headers, row)}
            return EchkinaReadyTable(**row)

        logger.debug(f"{self.name}: predict.")

        id_train = self.args.id_train
        date = self.args.date
        if id_train is None or date is None:
            print("Введите id насоса" if id_train is None else "",
                  "Введите дату, на которую нужно предсказать" if date else "")
            exit(0)

        max_date = date - datetime.timedelta(days=self.args.prediction_days)
        min_date = max_date - datetime.timedelta(days=self.args.analyze_days + 2)

        objects = []
        with open(self.args.file, 'r') as file:
            reader = csv.reader(file, delimiter=';')
            for row in reader:
                obj = create_obj(row)

                if id_train == obj.id_train and min_date <= obj.date <= max_date:
                    objects.append(obj)

        objects.sort(key=lambda x: x.date)
        dataloader = DataLoader(count_predictions_days=self.args.prediction_days,
                                count_analyze_days=self.args.analyze_days)

        if dataloader.get_len_batch() > len(objects):
            nice_print(text=f"Предсказание невозможно! Не хватает данных для предсказания. \n"
                            f"Требуется: {dataloader.get_len_batch()}\nИмеется только: {len(objects)}",
                       prefix='!', log_fun=logger.warning)
            return

        objects = objects[-dataloader.get_len_batch():]
        values = [obj.value for obj in objects]
        x = make_input_tensor(values)
        for model in self.__child_models.values():
            param1, param2 = None, None
            for obj in objects:
                if obj.unit == model.unit and obj.direction == model.direction:
                    param1 = obj.param1
                    param2 = obj.param2

            model.predict(x, param1, param2)

    def save(self, unit: Optional[int] = None, direction: Optional[int] = None) -> None:
        units = range(self.config.main.unit_start, self.config.main.unit_last + 1) if unit is None else [unit]
        directions = range(self.config.main.direction_start, self.config.main.direction_last + 1) \
            if direction is None else [direction]

        units_and_directions = ((unit, direction) for unit in units for direction in directions
                                if [unit, direction] not in self.config.main.off_unit_direction)

        nice_print(text=f"{self.name}: сохранение моделей с (unit, direction) {units_and_directions}.",
                   log_fun=logger.debug)
        for unit, direction in self.__child_models.keys():
            model = self.__child_models[self.key.format(unit, direction)]
            if model is not None:
                model.save()

    def __load(self) -> None:
        """
        Загрузка или создание моделей.
        """
        def create_new_model():
            logger.debug(text=f"Unit - {unit}, direction - {direction}: Создание модели {self.__model_type}")
            if self.__create_model_fun is None:
                print("Функция создания модели None. Модель не будет создана.")
                return

            model = self.__create_model_fun()
            optimizer = get_optimizer(model.parameters(), off_print=True)
            loss_function = get_loss_function(off_print=True)
            self.__child_models[self.key.format(unit, direction)] = MyChildModel(model=model,
                                                                                 optimizer=optimizer,
                                                                                 loss_fun=loss_function,
                                                                                 unit=unit, direction=direction)

        logger.info(text="Загрузка и создание моделей.")
        dictionary = self.config.main.checkpoint_save
        units_directions = [(unit, direction)
                            for unit in range(self.config.main.unit_start,
                                              self.config.main.unit_last + 1)
                            for direction in range(self.config.main.direction_start,
                                                   self.config.main.direction_last + 1)
                            if [unit, direction] not in self.config.main.off_unit_direction]

        for unit, direction in tqdm.tqdm(units_directions):
            try:
                nice_print(text=f"Unit - {unit}, direction - {direction}: Загрузка модели по пути "
                                f"{dictionary[self.key.format(unit, direction)]}", prefix='', postfix='-',
                           off=self.config.settings.off_all_prints, log_fun=logger.debug)
                self.__child_models[self.key.format(unit, direction)] = MyChildModel(
                    **torch.load(dictionary[self.key.format(unit, direction)]))
            except (KeyError, TypeError):
                create_new_model()
            except Exception as exp:
                print(f"{self.name}: Для юнита {unit} и направления {direction} модель не загружена. Ошибка: {exp}")
                create_new_model()
