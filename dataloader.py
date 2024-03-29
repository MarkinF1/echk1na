import yaml
import tqdm
import torch
import pickle
import os.path
import datetime
from typing import List, Optional, Tuple, Dict
from sklearn.model_selection import train_test_split

from logger import logger
from db_connection import DataBase
from db_classes import EchkinaReadyTableCrop
from supporting import nice_print, device, make_input_tensor, Config, database_name, Args


class DataLoader:
    """
    Загрузчик предназначенный для работы с данными.
    Реализован типо Singleton для отдельных дб. Таким образом,
    можно создавать несколько загрузчиков для разных баз данных.
    """
    class ArrayTypes:
        """
        3 типа массива (тренировочный, валидационный, тестировочный)
        """
        train = "train"
        validate = "validate"
        test = "test"

    __instance = {}
    __count_points = 2

    def __init__(self, count_predictions_days: int = 3, count_analyze_days: int = 10, count_directions: int = 3,
                 count_units: int = 3, database: str = database_name):
        if database not in self.__instance:
            self.__count_predictions_days = datetime.timedelta(count_predictions_days)
            self.__count_analyze_days = datetime.timedelta(count_analyze_days)

            self.__count_directions: int = count_directions
            self.__count_units: int = count_units

            self.__current_array_type: str = self.ArrayTypes.train
            self.__current_unit: Optional[int] = None
            self.__current_direction: Optional[int] = None
            self.__arrays: Dict[str, Tuple[EchkinaReadyTableCrop]] = {self.ArrayTypes.train: (),
                                                                      self.ArrayTypes.validate: (),
                                                                      self.ArrayTypes.test: ()}

            self.config = Config.getInstance()

            self.__database = DataBase(database=database)
            self.__instance[database] = self
            logger.info(f"Для DataLoader(database={database}) длина батча {self.get_len_batch()}.")

    @classmethod
    def getInstance(cls, database: str):
        try:
            return cls.__instance[database]
        except KeyError:
            logger.error(f"Dataloader для {database} не настроен.")
            return None

    def get_current_unit_direction(self) -> Tuple[Optional[int], Optional[int]]:
        """
        Возвращает текущие unit и direction, в которые установлен сейчас загрузчик.
        :return:
        """
        return self.__current_unit, self.__current_direction

    def is_unit_direction_valid(self) -> bool:
        """
        Проверка текущих unit и direction на валидность.
        :return:
        """
        return (self.__current_unit in range(Config.getInstance().main.unit_start,
                                             Config.getInstance().main.unit_last + 1)
                and self.__current_direction in range(Config.getInstance().main.direction_start,
                                                      Config.getInstance().main.direction_last + 1))

    @classmethod
    def __convert_days2count_values(cls, days: int) -> int:
        """
        Конвертирует дни в количество данных, которое там должно быть.
        :param days: дни
        :return: количество данных
        """
        # Использовал сначала это, но данные слишком расплывчаты, это не подойдет
        # in_one_day = self.__count_directions * self.__count_units * self.__count_points
        # count = int(days * 1.5) * in_one_day
        count = cls.__count_points * int(days * 1.5)
        return count

    @classmethod
    def get_len_batch(cls) -> int:
        """
        Возвращает длину батча необходимого для обучения.
        :return: длина батча
        """
        return cls.__convert_days2count_values(Args.getInstance().analyze_days)

    def set_unit_direction(self, unit: int, direction: int) -> None:
        """
        Для работы загрузчика необходимо установить ему конкретный unit
        и direction с помощью данной функции, после чего по нему можно будет проходиться.
        Здесь мы загружаем все строки с (unit, direction), которые подходят для подачи на модель,
        и разделяем их на три массива данных (train, valid, test) с коэффициентами из конфиг-файла.
        :param unit: unit
        :param direction: direction
        """
        logger.debug(f"Инициализация DataLoader'a с параметрами unit: {unit}, direction: {direction}")

        self.__current_unit = unit
        self.__current_direction = direction

        if not self.is_unit_direction_valid():
            print(f"Error: Dataloader не валидный. Установите unit и direction с помощью метода set_unit_direction")
            return

        del self.__arrays
        self.__arrays = {}
        arr = self.__database.get_ready_data_by_unit_direction_crop(unit=self.__current_unit,
                                                                    direction=self.__current_direction)
        valid_arr = []
        for elem in tqdm.tqdm(arr):
            objects = self.__database.get_ready_data_special(id_train=elem.id_train,
                                                             unit=self.__current_unit,
                                                             direction=self.__current_direction,
                                                             max_date=elem.date - self.__count_predictions_days,
                                                             min_date=elem.date - self.__count_predictions_days
                                                                                - self.__count_analyze_days
                                                                                - datetime.timedelta(days=2),
                                                             arr_idx=elem.arr_idx)
            if self.get_len_batch() > len(objects):
                continue

            objects = objects[-self.get_len_batch():]
            values = [obj.value for obj in objects]
            x = make_input_tensor(values)

            target = self.__database.get_ready_data_by_id(id_data=elem.id_)
            y = torch.tensor([target.value], dtype=torch.float32).to(device())

            valid_arr.append((x, y, target))

        self.__arrays[self.ArrayTypes.train], test_valid_arr = (
            train_test_split(valid_arr,
                             train_size=self.config.dataloader.train_size,
                             random_state=self.config.dataloader.random_state)
        )

        test_size = self.config.dataloader.test_size / (1.0 - self.config.dataloader.train_size)
        self.__arrays[self.ArrayTypes.validate], self.__arrays[self.ArrayTypes.test] = (
            train_test_split(test_valid_arr,
                             test_size=test_size,
                             random_state=self.config.dataloader.random_state)
        )

        logger.info(f"Валидных объектов: {len(valid_arr)}/{len(arr)}")

        logger.info(f"Тренировочных объектов: {len(self.__arrays[self.ArrayTypes.train])}/{len(valid_arr)}")

        logger.info(f"Валидационных объектов: {len(self.__arrays[self.ArrayTypes.validate])}/{len(valid_arr)}")

        logger.info(f"Тестовых объектов: {len(self.__arrays[self.ArrayTypes.test])}/{len(valid_arr)}")

    def save_pickle(self) -> None:
        """
        Сохранить дамп обработанных данных.
        Тип файла куда сохранить описан в конфиг-файле.
        """
        path = os.path.join(Config.getInstance().main.valid_objects_save_dir,
                            Config.getInstance().main.valid_objects_string)

        with open(path.format(
                self.config.dataloader.random_state, self.__count_analyze_days.days,
                self.__count_predictions_days.days, self.__current_unit, self.__current_direction
        ), "wb") as file:
            pickle.dump(self.__arrays, file)

    def load_pickle(self, unit: int, direction: int, path: str) -> None:
        """
        Заполнение загрузчика из файла.
        :param unit: unit
        :param direction: direction
        :param path: путь до дампа
        """
        self.__current_unit = unit
        self.__current_direction = direction

        with open(path.format(unit, direction), "rb") as file:
            self.__arrays = pickle.load(file)

    def check_train_id_is_valid(self, id_train: int) -> List[datetime.date]:
        """
        Проверка насоса по его id на валидные даты.
        :param id_train: id насоса
        :return: список валидных дат
        """
        objects = self.__database.get_ready_data_by_train(id_train=id_train)
        if len(objects) < 2:
            return []

        objects.sort(key=lambda x: (x.arr_idx, x.date))

        dd = {}
        for obj in objects:
            if obj.arr_idx not in dd:
                dd[obj.arr_idx] = []
            dd[obj.arr_idx].append(obj)

        dates = []
        for objs in dd.values():
            dates.append([])
            for obj in objs:
                dates[-1].append(datetime.date(year=obj.date.year, month=obj.date.month, day=obj.date.day))

        for i in range(len(dates) - 1, -1, -1):
            if len(dates[i]) < self.get_len_batch():
                dates.pop(i)

        result = []
        for date_period in dates:
            first_day = datetime.date(year=date_period[0].year, month=date_period[0].month, day=date_period[0].day)
            first_day += self.__count_analyze_days

            last_day = datetime.date(year=date_period[-1].year, month=date_period[-1].month, day=date_period[-1].day)
            last_day += self.__count_analyze_days

            current_day = first_day
            while current_day < last_day:
                result.append(current_day)
                current_day += datetime.timedelta(days=1)

        return result

    def __iter__(self):
        return self.__arrays[self.__current_array_type].__iter__()

    def __next__(self):
        return self.__arrays[self.__current_array_type].__next__()

    def __len__(self):
        return self.__arrays[self.__current_array_type].__len__()

    def train(self) -> None:
        """
        Переводит DataLoader в train режим
        """
        if self.__current_array_type != self.ArrayTypes.train:
            self.__current_array_type = self.ArrayTypes.train

    def validate(self) -> None:
        """
        Переводит DataLoader в validate режим
        """
        if self.__current_array_type != self.ArrayTypes.validate:
            self.__current_array_type = self.ArrayTypes.validate

    def eval(self):
        """
        Переводит DataLoader в test режим
        """
        if self.__current_array_type != self.ArrayTypes.test:
            self.__current_array_type = self.ArrayTypes.test


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Описание программы")

    parser.add_argument("-p", "--prediction_days", type=int, choices=[3, 14],
                        help="На сколько дней вперед вы хотите предсказать.")
    parser.add_argument("-a", "--analyze_days", type=int,
                        help="Сколько дней взять для анализа.")
    parser.add_argument("-c", "--config", type=str, help="Установка конкретного конфига программы.")
    parser.add_argument("-u", "--unit", type=int, default=None)
    parser.add_argument("-d", "--direction", type=int, default=None)

    args = parser.parse_args()
    with open(args.config, "r") as file:
        config = Config(yaml.load(file, yaml.SafeLoader))
    torch.cuda.empty_cache()

    dataloader = DataLoader(count_predictions_days=args.prediction_days,
                            count_analyze_days=args.analyze_days,
                            count_directions=3,
                            count_units=3)

    if args.unit is not None or args.direction is not None:
        if args.unit is not None and args.direction is not None:
            print(f"Unit: {args.unit}, Direction: {args.direction}")
            dataloader.set_unit_direction(args.unit, args.direction)
            dataloader.save_pickle()
        else:
            print("Нужно указать и unit, и direction, или же ничего из этого не указывать.")
            exit(-1)
    else:
        for unit in range(3):
            for direction in range(1, 4):
                print(f"Unit: {unit}, Direction: {direction}")
                path = os.path.join(Config.getInstance().main.valid_objects_save_dir,
                                    Config.getInstance().main.valid_objects_string).format(
                    config.dataloader.random_state, args.analyze_days, args.prediction_days, unit, direction)

                if os.path.exists(path):
                    print(f"Дамп {path}, direction {direction} уже существует, пропускаю.")
                    continue

                dataloader.set_unit_direction(unit, direction)
                dataloader.save_pickle()
