import datetime
import os.path
import pickle
from typing import List, Optional, Tuple, Dict

import torch
import tqdm
from sklearn.model_selection import train_test_split

from db_classes import EchkinaReadyTableCrop, EchkinaReadyTable
from db_connection import DataBase
from supporting import nice_print, device, make_input_tensor, Config, database_name


class DataLoader:
    class ArrayTypes:
        train = "train"
        validate = "validate"
        test = "test"

    __instance = {}

    def __init__(self, count_predictions_days: int = 3, count_analyze_days: int = 10, count_directions: int = 3,
                 count_units: int = 3, database: str = database_name):
        if database not in self.__instance:
            self.__count_predictions_days = datetime.timedelta(count_predictions_days)
            self.__count_analyze_days = datetime.timedelta(count_analyze_days)

            self.__count_directions: int = count_directions
            self.__count_units: int = count_units

            self.__count_points = 2
            self.__current_array_type: str = self.ArrayTypes.train
            self.__current_unit: Optional[int] = None
            self.__current_direction: Optional[int] = None
            self.__arrays: Dict[str, Tuple[EchkinaReadyTableCrop]] = {self.ArrayTypes.train: (),
                                                                      self.ArrayTypes.validate: (),
                                                                      self.ArrayTypes.test: ()}

            self.config = Config.getInstance()

            self.__database = DataBase(database=database)
            self.__instance[database] = self

    @classmethod
    def getInstance(cls, database: str):
        try:
            return cls.__instance[database]
        except KeyError:
            return None

    def get_current_unit_direction(self):
        return self.__current_unit, self.__current_direction

    def is_unit_direction_valid(self):
        return (self.__current_unit in range(Config.getInstance().main.unit_start,
                                             Config.getInstance().main.unit_last + 1)
                and self.__current_direction in range(Config.getInstance().main.direction_start,
                                                      Config.getInstance().main.direction_last + 1))

    def __convert_days2count_values(self, days: int) -> int:
        in_one_day = self.__count_directions * self.__count_units * self.__count_points
        count = int(days * 1.5) * in_one_day
        return count

    def get_len_batch(self) -> int:
        return self.__convert_days2count_values(self.__count_analyze_days.days)

    def set_unit_direction(self, unit: int, direction: int) -> None:
        nice_print(text=f"Инициализация DataLoader'a "
                        f"с параметрами unit: {unit}, direction: {direction}", suffix="=")

        self.__current_unit = unit
        self.__current_direction = direction

        if not self.is_unit_direction_valid():
            print(f"Error: Dataloader не валидный. Установите unit и direction с помощью метода set_unit_direction")
            return

        del self.__arrays
        arr = self.__database.get_ready_data_by_unit_direction_crop(unit=self.__current_unit,
                                                                    direction=self.__current_direction)
        valid_arr = []
        for elem in tqdm.tqdm(arr):
            objects = self.__database.get_ready_data_special(id_train=elem.id_train,
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

        nice_print(text=f"Валидных объектов: {len(valid_arr)}/{len(arr)}",
                   suffix='', suffix2='-')

        nice_print(text=f"Тренировочных объектов: {len(self.__arrays[self.ArrayTypes.train])}/{len(valid_arr)}",
                   suffix='', suffix2='-')

        nice_print(text=f"Валидационных объектов: {len(self.__arrays[self.ArrayTypes.validate])}/{len(valid_arr)}",
                   suffix='', suffix2='-')

        nice_print(text=f"Тестовых объектов: {len(self.__arrays[self.ArrayTypes.test])}/{len(valid_arr)}",
                   suffix='', suffix2='-')

    def save_pickle(self):
        path = os.path.join(Config.getInstance().main.valid_objects_save_dir,
                            Config.getInstance().main.valid_objects_string)

        with open(path.format(
                self.config.dataloader.random_state, self.__count_analyze_days.days,
                self.__count_predictions_days.days, self.__current_unit, self.__current_direction
        ), "wb") as file:
            pickle.dump(self.__arrays, file)

    def load_pickle(self, unit, direction, path):
        self.__current_unit = unit
        self.__current_direction = direction

        with open(path.format(unit, direction), "rb") as file:
            self.__arrays = pickle.load(file)

    def check_train_id_is_valid(self, id_train: int) -> List[datetime.date]:
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

    def train(self) -> None:
        """
        Переводит DataLoader в train режим:
        меняет массив на тренировочный, обнуляет индексацию.
        """
        if self.__current_array_type != self.ArrayTypes.train:
            self.__current_array_type = self.ArrayTypes.train

    def validate(self) -> None:
        """
        Переводит DataLoader в validate режим:
        меняет массив на валидационный, обнуляет индексацию.
        """
        if self.__current_array_type != self.ArrayTypes.validate:
            self.__current_array_type = self.ArrayTypes.validate

    def eval(self):
        """
        Переводит DataLoader в test режим:
        меняет массив на тестовый, обнуляет индексацию.
        """
        if self.__current_array_type != self.ArrayTypes.test:
            self.__current_array_type = self.ArrayTypes.test


if __name__ == "__main__":
    dataloader = DataLoader()
    dataloader.check_train_id_is_valid(1613)
