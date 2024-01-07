import os
import pickle
from time import time
from typing import Optional, Dict, List

import torch
import tqdm
import wandb

from dataloader import DataLoader
from supporting import Config, nice_print, get_time, Args, get_optimizer, get_loss_function, save_obj, device, \
    database_name


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
                nice_print(text=f"Нашел дамп валидных объектов для unit {self.unit} и direction {self.direction}.",
                           suffix='', suffix2='-')
                print("Загружаю дамп...", end="\r")
                dataloader.load_pickle(self.unit, self.direction, path)
                print("Дамп загружен.\n", end="\r")
            else:
                nice_print(text=f"Дамп валидных объектов для analyze {self.args.analyze_days}, prediction "
                                f"{self.args.prediction_days}, unit {self.unit} и direction {self.direction} не найден."
                                f" Dataloader устанавливает массивы для обучения.", suffix='', suffix2='-')
                dataloader.set_unit_direction(self.unit, self.direction)
                dataloader.save_pickle()
        else:
            nice_print(text=f"Дамп валидных объектов для unit {self.unit} и direction {self.direction} запрещен к "
                            f"загрузке. Запускаюсь с пустым массивом валидных объектов.", suffix='', suffix2='-')
            dataloader.set_unit_direction(self.unit, self.direction)
            dataloader.save_pickle()
        return dataloader

    def train(self) -> None:
        valid_save_path = os.path.join(self.config.main.valid_objects_save_dir, self.config.main.valid_objects_string)
        valid_save_path = valid_save_path.format(self.args.analyze_days, self.args.prediction_days, "{0}", "{1}")

        nice_print(text=f"ОБУЧЕНИЕ ДЛЯ UNIT: {self.unit}, DIRECTION: {self.direction}",
                   suffix="/\\", suffix2="\\/", num=17)

        train_arr = self.__load_valid_objects(path=valid_save_path)

        # Настройка вспомогательных параметров
        epoch_time = time()
        if self.config.wandb.turn_on:
            graphic_wandb = wandb.init(
                          name=self.config.wandb.name.format(self.unit, self.direction),
                          project=self.config.wandb.project,
                          config={"epochs": self.config.model.epoch, "lr": self.config.model.lr})

        nice_print(text="Начинаю обучение.", suffix='*', off=self.config.settings.off_all_prints)
        start_epoch = self.epoch
        for self.epoch in range(start_epoch, self.config.model.epoch):
            self.model.train()

            if self.config.settings.print_time:
                epoch_duration = time() - epoch_time
                nice_print(text=f"Epoch: {self.epoch + 1}/{self.config.model.epoch} " +
                                (f"Время одной эпохи: {get_time(epoch_duration)}" if self.epoch else ""),
                           suffix="=", off=self.config.settings.off_all_prints)

                epoch_time = time()

            sum_loss = 0
            count = 0
            for i, (x, y, target) in enumerate(tqdm.tqdm(train_arr)):
                count += 1
                self.optim.zero_grad()
                output = self.model(x, target.param1, target.param2)

                loss = self.loss_fun(output, y)
                loss.backward()
                self.optim.step()

                if self.config.settings.print_predict and i % self.config.settings.print_predict_step == 0:
                    print(f"model.predict: {output}\ntarget:{y}\nloss: {loss.item()}")

                sum_loss += loss.item()

            if not count:
                print(f"Не нашел валидных объектов для unit {self.unit}, direction {self.direction}. "
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
                    print(f"[ERROR] Wandb: не смог сделать log. {exp}.")

            self.validate()

            if self.epoch % self.config.model.lr_step == (self.config.main.lr_step - 1):
                for g in self.optim.param_groups:
                    g["lr"] *= self.config.model.lr_decay

        if self.config.wandb.turn_on:
            try:
                graphic_wandb.finish()
            except Exception:
                print("Не удалось нормально завершить wandb_unit_%s_direction_%s" % (self.unit, self.direction))

    def validate(self) -> None:
        self.model.eval()
        dataloader = DataLoader.getInstance(database=database_name)
        dataloader.validate()

        with torch.no_grad():
            if self.valid_validate_objects:
                validate_arr = self.valid_validate_objects
                is_valid = True
            else:
                validate_arr = dataloader
                is_valid = False

            sum_loss = 0
            count = 0
            for x, y, objects, target, i, num in validate_arr:
                count += 1
                output = self.model(x, target.param1, target.param2)
                loss = self.loss_fun(output, y)

                if self.config.settings.print_predict and i % self.config.settings.print_predict_step == 0:
                    print(f"model.predict: {output}\ntarget:{y}\nloss: {loss.item()}")

                sum_loss += loss.item()

                if not is_valid:
                    self.valid_validate_objects.append((x, y, objects, target, i, num))

                print(f"Валидация: [{i}/{num}]", end='\r')

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
                nice_print(text=f"Best_loss: {self.best_loss} ----- Current_loss: {epoch_loss} "
                                f"Save checkpoint epoch: {self.epoch}",
                           suffix="-")
                self.best_loss = epoch_loss
                self.save()

    def eval(self) -> None:
        self.model.eval()
        dataloader = DataLoader.getInstance(database=database_name)
        dataloader.set_unit_direction(self.unit, self.direction)
        dataloader.eval()

        with torch.no_grad():
            test_arr = dataloader

            losses = []
            count = 0
            for x, y, objects, target, i, num in test_arr:
                count += 1
                output = self.model(x, target.param1, target.param2)
                loss = self.loss_fun(output, y)
                loss.backward()
                losses.append(loss.item())

            epoch_loss = sum(losses) / count

        nice_print(text=f"Ошибка на тестовом наборе: {epoch_loss}.\nМаксимальная ошибка: {max(losses)}.\n"
                        f"Минимальная ошибка: {min(losses)}.", suffix='=')

    def predict(self) -> None:
        pass

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
    def __init__(self, create_model_fun):
        self.__create_model_fun = create_model_fun
        self.__model_type = self.__create_model_fun().__class__.__name__

        self.args = Args.getInstance()
        self.config = Config.getInstance()
        self.name = self.__class__.__name__
        self.key = "({0}, {1})"

        self.__child_models: Dict[str, MyChildModel] = {}
        self.__load()

    def train(self):
        nice_print(f"{self.name}: train.")
        for model in self.__child_models.values():
            model.train()

    def eval(self):
        nice_print(f"{self.name}: eval.")
        for model in self.__child_models.values():
            model.eval()

    def predict(self):
        nice_print(f"{self.name}: predict.")
        for model in self.__child_models.values():
            model.predict()

    def save(self, unit: Optional[int] = None, direction: Optional[int] = None) -> None:
        units = range(self.config.main.unit_start, self.config.main.unit_last + 1) if unit is None else [unit]
        directions = range(self.config.main.direction_start, self.config.main.direction_last + 1) \
            if direction is None else [direction]

        units_and_directions = ((unit, direction) for unit in units for direction in directions
                                if [unit, direction] not in self.config.main.off_unit_direction)

        nice_print(f"{self.name}: save models with (unit, direction) {units_and_directions}.")
        for unit, direction in self.__child_models.keys():
            model = self.__child_models[self.key.format(unit, direction)]
            if model is not None:
                model.save()

    def __load(self) -> None:
        """
        Загрузка или создание моделей.
        """
        def create_new_model():
            nice_print(text=f"Unit - {unit}, direction - {direction}: Создание модели {self.__model_type}",
                       suffix='', suffix2='-', off=self.config.settings.off_all_prints)
            model = self.__create_model_fun()
            optimizer = get_optimizer(model.parameters(), off_print=True)
            loss_function = get_loss_function(off_print=True)
            self.__child_models[self.key.format(unit, direction)] = MyChildModel(model=model,
                                                                                 optimizer=optimizer,
                                                                                 loss_fun=loss_function,
                                                                                 unit=unit, direction=direction)

        nice_print(text="Загрузка и создание моделей.")
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
                                f"{dictionary[self.key.format(unit, direction)]}", suffix='', suffix2='-',
                           off=self.config.settings.off_all_prints)
                self.__child_models[self.key.format(unit, direction)] = MyChildModel(
                    **torch.load(dictionary[self.key.format(unit, direction)]))
            except (KeyError, TypeError):
                create_new_model()
            except Exception as exp:
                print(f"{self.name}: Для юнита {unit} и направления {direction} модель не загружена. Ошибка: {exp}")
                create_new_model()
