import os
import csv
import tqdm
import torch
#import wandb
import datetime
from typing import Optional, Dict, List

from db_connection import DataBase
from logger import logger
from dataloader import DataLoader
from db_classes import EchkinaReadyTable
from supporting import Config, nice_print, Args, get_optimizer, get_loss_function, save_obj, database_name, \
    make_input_tensor, str2date


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
                logger.info(f"Нашел дамп валидных объектов для unit {self.unit} и direction {self.direction}.")
                print("Загружаю дамп...", end="\r")
                dataloader.load_pickle(self.unit, self.direction, path)
                print("Дамп загружен.\n", end="\r")
            else:
                logger.info(f"Дамп валидных объектов для analyze {self.args.analyze_days}, prediction "
                            f"{self.args.prediction_days}, unit {self.unit} и direction {self.direction} не найден."
                            f" Dataloader устанавливает массивы для обучения.")
                dataloader.set_unit_direction(self.unit, self.direction)
                dataloader.save_pickle()
        else:
            logger.info(f"Дамп валидных объектов для unit {self.unit} и direction {self.direction} запрещен к "
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
        if self.config.wandb.turn_on:
            model_type = self.config.model.tp
            model_param_string = '_'.join(list(map(str, self.config.__getattribute__(model_type)._asdict().values())))
            graphic_wandb = wandb.init(
                          name=self.config.wandb.name.format(model_type, model_param_string,
                                                             self.config.model.optimizer, self.args.analyze_days,
                                                             self.args.prediction_days),
                          project=self.config.wandb.project.format(self.unit, self.direction),
                          config={"epochs": self.config.model.epoch, "lr": self.config.model.lr})

        logger.info("Начинаю обучение.")
        start_epoch = self.epoch
        logger.info("Эпоха: ")
        for epoch in tqdm.tqdm(range(start_epoch, self.config.model.epoch)):
            self.epoch = epoch
            self.model.train()
            train_arr.train()
            epoch_loss = 0
            count_inf_data = 0

            for i, (x, y, target) in enumerate(train_arr):
                self.optim.zero_grad()

                output = self.model(x, target.param1, target.param2)
                if output is None:
                    count_inf_data += 1
                    continue

                loss = self.loss_fun(output, y)
                loss.backward()
                self.optim.step()

                if self.config.settings.print_predict and i % self.config.settings.print_predict_step == 0:
                    logger.debug(f"model.predict: {output}\ntarget:{y}\nloss: {loss.item()}")

                epoch_loss += loss.item() / len(train_arr)
                if epoch_loss is None:
                    input()

            if count_inf_data and self.epoch == start_epoch:
                logger.debug(f"Train: Найдено {count_inf_data} тензора с бесконечными значениями из {len(train_arr)}.")

            if not len(train_arr):
                logger.warning(f"Не нашел валидных объектов для unit {self.unit}, direction {self.direction}. "
                               f"Продолжить [y/n]?")
                n = input()
                if n == 'y':
                    break
                else:
                    exit(-1)

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
            epoch_loss = 0
            count_inf_data = 0
            for i, (x, y, target) in enumerate(dataloader):
                output = self.model(x, target.param1, target.param2)
                if output is None:
                    count_inf_data += 1
                    continue

                loss = self.loss_fun(output, y)

                if self.config.settings.print_predict and i % self.config.settings.print_predict_step == 0:
                    print(f"model.predict: {output}\ntarget:{y}\nloss: {loss.item()}")

                epoch_loss += loss.item() / len(dataloader)

            if count_inf_data and not self.epoch:
                logger.debug(f"Validate: Найдено {count_inf_data} тензора с бесконечными "
                             f"значениями из {len(dataloader)}.")

            if not len(dataloader):
                print(f"Не нашел валидных объектов для unit {self.unit}, direction {self.direction}. "
                      f"Продолжить [y/n]?")
                n = input()
                if n == 'y':
                    return
                else:
                    exit(-1)

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
            count_inf_data = 0
            for x, y, objects, target, i, num in tqdm.tqdm(dataloader):
                output = self.model(x, target.param1, target.param2)
                if output is None:
                    count_inf_data += 1
                    continue

                loss = self.loss_fun(output, y)
                loss.backward()
                losses.append(loss.item())

            if count_inf_data:
                logger.debug(f"Test: Найдено {count_inf_data} тензора с бесконечными значениями из {len(dataloader)}.")

            epoch_loss = sum(losses) / len(dataloader)

        nice_print(text=f"Ошибка на тестовом наборе: {epoch_loss}.\nМаксимальная ошибка: {max(losses)}.\n"
                        f"Минимальная ошибка: {min(losses)}.", prefix='=', log_fun=logger.info)

    def predict(self, x: torch.Tensor, param1, param2, alarm3 = None, alarm4 = None) -> None:
        y: torch.Tensor = self.model(x, param1, param2)
        if y is None:
            logger.warning(f"Входные данные содержат ошибку! Tensor: {x}")
            return

        y = y[0]

        status = "в пределах нормы"
        if alarm3 is not None and y > alarm3:
            status = "превышение по аларму 3 ({})".format(alarm3)
        if alarm4 is not None and y > alarm4:
            status = "превышение по аларму 4 ({})".format(alarm4)

        logger.info(f"Unit: {self.unit}, direction: {self.direction}, предсказание: {y}, статус: {status}.")

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
        id_train = self.args.id_train
        date = self.args.date
        if id_train is None or date is None:
            print("Введите id насоса" if id_train is None else "",
                  "Введите дату, на которую нужно предсказать" if date else "")
            exit(0)

        variants = {"csv": self.__predict_csv,
                    "db": self.__predict_database
                    }
        variants[self.args.type_]()

    def __predict_csv(self):
        def create_obj(row):
            headers = ["id", "id_train", "id_point", "id_measure", "direction", "unit", "date", "value",
                       "alarm3", "alarm4", "param1", "param2", "arr_idx"]
            for i in range(len(row)):
                if row[i] == "":
                    row[i] = None
                elif row[i].find(".") != -1:
                    row[i] = float(row[i])
                elif len(row[i].split("-")) == 3:
                    row[i] = str2date(row[i].split(' ')[0])
                else:
                    try:
                        row[i] = int(row[i])
                    except Exception as exp:
                        logger.exception(f"Не смог преобразовать {type(row[i])} в int. Exception: {exp}")
            row = {key: val for key, val in zip(headers, row)}
            return EchkinaReadyTable(**row)

        logger.info(f"{self.name}: predict from csv file.")
        id_train = self.args.id_train
        date = self.args.date

        max_date = date - datetime.timedelta(days=self.args.prediction_days)
        min_date = max_date - datetime.timedelta(days=self.args.analyze_days + 2)

        objects = []
        with open(self.args.file, 'r') as file:
            reader = csv.reader(file, delimiter=';')
            for row in reader:
                obj = create_obj(row)

                if id_train == obj.id_train:
                    if min_date <= obj.date <= max_date:
                        objects.append(obj)

        objects.sort(key=lambda x: x.date)
        dataloader = DataLoader(count_predictions_days=self.args.prediction_days,
                                count_analyze_days=self.args.analyze_days)

        for model in self.__child_models.values():
            objects_unit_direction = [obj for obj in objects
                                      if obj.unit == model.unit and obj.direction == model.direction]
            if dataloader.get_len_batch() > len(objects_unit_direction):
                nice_print(text=f"Предсказание для Unit - {model.unit}, Direction - {model.direction} невозможно! "
                                f"Не хватает данных для предсказания.\n"
                                f"Требуется: {dataloader.get_len_batch()}\nИмеется только: {len(objects_unit_direction)}",
                           prefix='!', log_fun=logger.warning)
                continue

            objects_unit_direction = objects_unit_direction[-dataloader.get_len_batch():]
            values = [obj.value for obj in objects_unit_direction]
            x = make_input_tensor(values)

            param1, param2, alarm3, alarm4 = None, None, None, None
            for obj in objects_unit_direction:
                if obj.unit == model.unit and obj.direction == model.direction:
                    if param1 is None:
                        param1 = obj.param1
                    if param2 is None:
                        param2 = obj.param2
                    if alarm3 is None:
                        alarm3 = obj.alarm3
                    if alarm4 is None:
                        alarm4 = obj.alarm4
                if (param1 is not None and param2 is not None
                        and alarm3 is not None and alarm4 is not None):
                    break

            model.predict(x, param1, param2, alarm3, alarm4)

    def __predict_database(self):
        logger.debug(f"{self.name}: predict from database {self.args.base}.")
        id_train = self.args.id_train
        date = self.args.date

        max_date = date - datetime.timedelta(days=self.args.prediction_days)
        min_date = max_date - datetime.timedelta(days=self.args.analyze_days + 2)

        dataloader = DataLoader(count_predictions_days=self.args.prediction_days,
                                count_analyze_days=self.args.analyze_days)

        for model in self.__child_models.values():
            objects = DataBase(database=self.args.base).get_ready_data_special(id_train=id_train,
                                                                               unit=model.unit,
                                                                               direction=model.direction,
                                                                               max_date=max_date,
                                                                               min_date=min_date)
            objects.sort(key=lambda x: x.date)
            if dataloader.get_len_batch() > len(objects):
                nice_print(text=f"Предсказание для Unit - {model.unit}, Direction - {model.direction} невозможно! "
                                f"Не хватает данных для предсказания.\n"
                                f"Требуется: {dataloader.get_len_batch()}\nИмеется только: {len(objects)}",
                           prefix='!', log_fun=logger.warning)
                continue

            objects = objects[-dataloader.get_len_batch():]
            values = [obj.value for obj in objects]
            x = make_input_tensor(values)

            param1, param2, alarm3, alarm4 = None, None, None, None
            for obj in objects:
                if obj.unit == model.unit and obj.direction == model.direction:
                    if param1 is None:
                        param1 = obj.param1
                    if param2 is None:
                        param2 = obj.param2
                    if alarm3 is None:
                        alarm3 = obj.alarm3
                    if alarm4 is None:
                        alarm4 = obj.alarm4
                if (param1 is not None and param2 is not None
                        and alarm3 is not None and alarm4 is not None):
                    break

            model.predict(x, param1, param2, alarm3, alarm4)

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
            logger.debug(f"Unit - {unit}, direction - {direction}: Создание модели {self.__model_type}")
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

        logger.info("Загрузка и создание моделей.")
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
