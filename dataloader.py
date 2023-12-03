import datetime
import os
from typing import List

from sklearn.model_selection import train_test_split
from sqlalchemy.engine import Row
from tqdm import tqdm

from db_classes import EchkinaReadyTableCrop
from db_connection import DataBase
from yamlcreator import YamlCreator


class DataLoader:
    def __init__(self, count_predictions_days: int = 3, count_analyze_days: int = 10, count_directions: int = 3,
                 count_units: int = 3, count_points: int = 2):
        self.__count_predictions_days: int = count_predictions_days
        self.__count_analyze_days: int = count_analyze_days

        self.__count_directions: int = count_directions
        self.__count_units: int = count_units
        self.__count_points: int = count_points

        self.__is_train: bool = True
        self.__train_arr: List[EchkinaReadyTableCrop] = []
        self.__test_arr: List[EchkinaReadyTableCrop] = []
        self.__current_arr: List[EchkinaReadyTableCrop] = []
        self.__valid_arr: List[bool] = []
        self.__database = DataBase()
        self.__idx = 0

        self.__init()

    def __init(self) -> None:
        self.__init_valid_array()

        crop_valid_objects = list(map(self.__database.get_ready_data_by_id_train_date, self.__valid_arr))
        self.__train_arr, self.__test_arr = train_test_split(crop_valid_objects, test_size=0.2)
        self.__current_arr = self.__train_arr

    def __init_valid_array(self):
        if not os.path.exists("./checkpoint.yml"):
            ids = self.__database.get_ready_data_all_id()
            self.__valid_arr = []
            print("Проверка валидности всех насосов. "
                  "Выбираю насосы, которые можно анализировать на основе предыдущих показаний.")
            for id_ in tqdm(ids):
                if self.__valid_check(id_obj=id_):
                    self.__valid_arr.append(id_)
            print(f"Количество валидных насосов, которые будут использоваться "
                  f"для тренировки и теста в качестве таргета: {len(self.__valid_arr)}/{len(ids)}\n"
                  f"Количество насосов, которые будут использоваться исключительно "
                  f"в качестве X значения в тренировках и тестах: {len(ids) - len(self.__valid_arr)}/{len(ids)}")
            valid_file = YamlCreator()
            valid_file.add_parameter(name="valid_arr", value=self.__valid_arr, description="Валидный массив ids")
            valid_file.save(path="./", filename="checkpoint.yml")
        else:
            return NotImplementedError

    def __valid_check(self, id_obj) -> bool:
        obj = self.__database.get_ready_data_by_id(id_data=id_obj)
        predict_days = datetime.timedelta(self.__count_predictions_days)
        result = self.__database.get_ready_data_special(id_train=obj.id_train,
                                                        max_date=obj.date - predict_days,
                                                        arr_idx=obj.arr_idx)
        return len(result) >= self.get_len_batch()

    def get_len_batch(self):
        return self.__convert_days2count_values(self.__count_analyze_days)

    def __convert_days2count_values(self, days: int) -> int:
        in_one_day = self.__count_directions * self.__count_units * self.__count_points
        count = int(days * 1.5) * in_one_day
        return count

    def __iter__(self):
        while self.__idx < len(self.__current_arr):
            crop_obj = self.__current_arr[self.__idx]
            self.__idx += 1
            predict_days = datetime.timedelta(days=self.__count_predictions_days)
            objects = self.__database.get_ready_data_special(id_train=crop_obj.id_train,
                                                             max_date=crop_obj.date - predict_days,
                                                             arr_idx=crop_obj.arr_idx)
            # yield item

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

    for id in dataloader:
        print(id)
