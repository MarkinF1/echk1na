import datetime
import time
from typing import List, Optional, Tuple

import torch
from torchvision import transforms
from sklearn.model_selection import train_test_split

from db_classes import EchkinaReadyTableCrop, EchkinaReadyTable
from db_connection import DataBase
from supporting import nice_print


class DataLoader:
    def __init__(self, count_predictions_days: int = 3, count_analyze_days: int = 10, count_directions: int = 3,
                 count_units: int = 3, count_points: int = 2):
        self.__count_predictions_days: int = count_predictions_days
        self.__count_analyze_days: int = count_analyze_days

        self.__count_directions: int = count_directions
        self.__count_units: int = count_units
        self.__count_points: int = count_points

        self.__idx = 0
        self.__is_train: bool = True
        self.__train_arr: List[EchkinaReadyTableCrop] = []
        self.__test_arr: List[EchkinaReadyTableCrop] = []
        self.__current_arr: List[EchkinaReadyTableCrop] = []
        self.__current_unit: Optional[int] = None
        self.__current_direction: Optional[int] = None

        self.__database = DataBase()

    def get_current_unit_direction(self):
        return self.__current_unit, self.__current_direction

    def is_unit_direction_valid(self):
        return (self.__current_unit in range(self.__count_units) and
                self.__current_direction in range(1, self.__count_directions))

    def set_unit_direction(self, unit: int, direction: int) -> None:
        nice_print(text=f"Инициализация DataLoader'a "
                        f"с параметрами unit: {unit}, direction: {direction}", suffix="=")

        self.__current_unit = unit
        self.__current_direction = direction

        print(f"Unit: {self.__current_unit + 1}/{self.__count_units}, "
              f"Direction: {self.__current_direction}/{self.__count_directions}")
        arr = self.__database.get_ready_data_by_unit_direction_crop(unit=self.__current_unit,
                                                                    direction=self.__current_direction)
        self.__train_arr, self.__test_arr = train_test_split(arr, test_size=0.2, random_state=25)

        self.__current_arr = self.__train_arr if self.__is_train else self.__test_arr

    def __valid_check(self, id_obj: int) -> bool:
        obj = self.__database.get_ready_data_by_id(id_data=id_obj)
        predict_days = datetime.timedelta(self.__count_predictions_days)
        result = self.__database.get_ready_data_special(id_train=obj.id_train,
                                                        max_date=obj.date - predict_days,
                                                        arr_idx=obj.arr_idx)
        return len(result) >= self.get_len_batch()

    def get_len_batch(self) -> int:
        return self.__convert_days2count_values(self.__count_analyze_days)

    def __convert_days2count_values(self, days: int) -> int:
        in_one_day = self.__count_directions * self.__count_units * self.__count_points
        count = int(days * 1.5) * in_one_day
        return count

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
            first_day += datetime.timedelta(days=self.__count_analyze_days)

            last_day = datetime.date(year=date_period[-1].year, month=date_period[-1].month, day=date_period[-1].day)
            last_day += datetime.timedelta(days=self.__count_analyze_days)

            current_day = first_day
            while current_day < last_day:
                result.append(current_day)
                current_day += datetime.timedelta(days=1)

        return result

    def get_by_unit_direction(self, unit, direction) -> Tuple[torch.tensor,
                                                              torch.tensor,
                                                              List[EchkinaReadyTable],
                                                              EchkinaReadyTable, int, int]:
        self.set_unit_direction(unit, direction)

        if not self.is_unit_direction_valid():
            print(f"Dataloader не валидный. Установите unit и direction с помощью метода set_unit_direction")
            return None, None

        self.__idx = 0
        while self.__idx < len(self.__current_arr):
            crop_obj = self.__current_arr[self.__idx]
            self.__idx += 1
            predict_days = datetime.timedelta(days=self.__count_predictions_days)
            objects = self.__database.get_ready_data_special(id_train=crop_obj.id_train,
                                                             max_date=crop_obj.date - predict_days,
                                                             arr_idx=crop_obj.arr_idx)
            if self.get_len_batch() > len(objects):
                continue

            # values = [[], []]
            # for obj in objects[-self.get_len_batch():]:
            #     values[0].append(obj.value)
            #     values[1].append(time.mktime(obj.date.timetuple()))
            objects = objects[-self.get_len_batch():]
            values = [obj.value for obj in objects]
            # values = [[], []]
            # for i, val in enumerate(vals):
            #     values[i % 2].append(val)
            # values = [values[:int(len(values) // 2)], values[int(len(values)) // 2:]]
            target = self.__database.get_ready_data_by_id(id_data=crop_obj.id_)

            x = torch.tensor(values, dtype=torch.float32)
            mean = torch.mean(x)
            std = torch.std(x)
            x = (x - mean) / std

            y = torch.tensor([target.value], dtype=torch.float32)
            yield x, y, objects, target, self.__idx, len(self.__current_arr)

    def train(self) -> None:
        """
        Переводит DataLoader в train режим:
        меняет массив на тренировочный, обнуляет индексацию.
        """
        if not self.__is_train:
            self.__is_train = True
            self.__idx = 0
            self.__current_arr = self.__train_arr

    def eval(self):
        """
        Переводит DataLoader в test режим:
        меняет массив на тестовый, обнуляет индексацию.
        """
        if self.__is_train:
            self.__is_train = False
            self.__idx = 0
            self.__current_arr = self.__test_arr


if __name__ == "__main__":
    dataloader = DataLoader()
    dataloader.check_train_id_is_valid(1613)
