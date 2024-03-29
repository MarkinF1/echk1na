import time
from copy import copy
import numpy as np

from scipy import interpolate
import os
import re
from collections import namedtuple
from datetime import datetime, timedelta
from random import shuffle
from typing import Optional, List

from tqdm import tqdm
from yaml import load, SafeLoader

from db_classes import EchkinaReadyTable
from db_connection import DataBase
from yamlcreator import YamlCreator

_CONFIG = namedtuple('Config', ("dataset_path",
                                'data_path', 'new_data_path',
                                "info_train_path", "new_info_train_path",
                                "info_points_path", "new_info_points_path",
                                "info_measures_path", "new_info_measures_path"))
with open("config.yml", 'r') as config_file:
    CONFIG = _CONFIG(**load(config_file, SafeLoader))


def delete_duplicate_rows(tables):
    for old_path, new_path in tables:
        with open(old_path, 'r') as old_file:
            with open(new_path, 'w+') as new_file:
                line = old_file.readline()
                while line:
                    new_file.write(line)
                    new_line = old_file.readline()
                    while new_line == line:
                        new_line = old_file.readline()
                    line = new_line


def delete_columns_data(old_path, new_path):
    with open(old_path, 'r') as data_file:
        new_data_file = open(new_path, 'w+')
        line = data_file.readline()
        while line:
            words = line.split(';')
            line = ';'.join(words[:-1]) + '\n'
            new_data_file.write(line)
            line = data_file.readline()


def xlsx2csv():
    import pandas

    file = pandas.read_excel("/home/user/Загрузки/Pumps_Struct_All_Trains.xlsx")
    file.to_csv(f"{CONFIG.dataset_path}/{CONFIG.info_train_path}", index=False, sep=';')
    file = pandas.read_excel("/home/user/Загрузки/Pumps_Struct_All_Points.xlsx")
    file.to_csv(f"{CONFIG.dataset_path}/{CONFIG.info_points_path}", index=False, sep=';')
    file = pandas.read_excel("/home/user/Загрузки/Pumps_Struct_All_Measures.xlsx")
    file.to_csv(f"{CONFIG.dataset_path}/{CONFIG.info_measures_path}", index=False, sep=';')


def push_data_to_db():
    from db_classes import EchkinaTrain, EchkinaPoint, EchkinaMeasure, EchkinaData, engine
    from sqlalchemy.orm import Session
    from sqlalchemy.exc import IntegrityError

    session = Session(bind=engine)
    train = False
    points = False
    measure = False
    data = False
    # train = True
    # points = True
    # measure = True
    data = True

    if train:
        columns = ["id_train", "name", "description"]
        types = [int, str, str]
        with open(os.path.join(CONFIG.dataset_path, CONFIG.new_info_train_path), 'r') as train_file:
            train_file.readline()
            line = train_file.readline()
            while line:
                lines = line.split(';')
                dd = {}
                for column, value, type_col in zip(columns, lines, types):
                    if value == 'NULL':
                        dd[column] = None
                    else:
                        try:
                            dd[column] = type_col(value)
                        except ValueError:
                            dd[column] = None
                try:
                    obj = EchkinaTrain(**dd)
                    session.add(obj)
                    session.commit()
                except Exception:
                    pass
                line = train_file.readline()

    if points:
        columns = ["id_train", "idPoint", "name", "description", "direction", "controllParametrType"]
        types = [int, int, str, str, int, int]
        with open(os.path.join(CONFIG.dataset_path, CONFIG.new_info_points_path), 'r') as points_file:
            points_file.readline()
            line = points_file.readline()
            while line:
                lines = line.split(';')
                dd = {}
                for column, value, type_col in zip(columns, lines, types):
                    if value == 'NULL':
                        dd[column] = None
                    else:
                        try:
                            dd[column] = type_col(value)
                        except ValueError:
                            dd[column] = None
                try:
                    obj = EchkinaPoint(**dd)
                    session.add(obj)
                    session.commit()
                except Exception:
                    pass
                line = points_file.readline()

    if measure:
        columns = ["idPoint", "idMeasure", "name", "description", "type_", "rangeType", "units", "param1", "param2",
                   "param3", "alarmType", "alarmLevel2", "alarmLevel3", "alarmLevel4"]
        types = [int, int, str, str, int, int, int, int, int, int, bool, float, float, float]
        with open(os.path.join(CONFIG.dataset_path, CONFIG.new_info_measures_path), 'r') as measure_file:
            measure_file.readline()
            line = measure_file.readline()
            while line:
                lines = line.split(';')
                dd = {}
                for column, value, type_col in zip(columns, lines, types):
                    if value == 'NULL':
                        dd[column] = None
                    else:
                        try:
                            dd[column] = type_col(value)
                        except ValueError:
                            dd[column] = None
                try:
                    obj = EchkinaMeasure(**dd)
                    session.add(obj)
                    session.commit()
                except Exception:
                    pass
                line = measure_file.readline()

    if data:
        with open(os.path.join(CONFIG.dataset_path, CONFIG.new_data_path), 'r') as data_file:
            data_file.readline()
            line = data_file.readline()
            while line:
                lines = line.split(';')
                try:
                    p = re.compile(r"\d+")
                    res = p.findall(lines[1])
                    res = list(map(int, res))
                    spisok = ["year", "month", "day", "hour", "minute", "second"]
                    res = {k: v for k, v in zip(spisok, res[:-1])}
                    d = datetime(**res)
                    obj = EchkinaData(idMeasure=int(lines[0]),
                                      date=d,
                                      value1=float(lines[2].replace(',', '.')))
                    session.add(obj)
                    session.commit()
                except Exception as exp:
                    print(exp)
                line = data_file.readline()


def delete_old_data_from_db():
    from db_connection import DataBase

    dataset = DataBase()
    for data_object in dataset.get_data_all():
        d = data_object.date
        if d.year < 2020:
            dataset.remove_data_by_id(data_object.id)


def delete_old_measures_from_db():
    from db_connection import DataBase
    from tqdm import tqdm

    dataset = DataBase()
    for measure in tqdm(dataset.get_measure_all()):
        if not dataset.get_data_by_measure(measure.idMeasure):
            dataset.remove_measure_by_id(id_measure=measure.idMeasure)


def add_type_of_class_data():
    from db_connection import DataBase
    from tqdm import tqdm

    dataset = DataBase()
    dd = {}
    for measure_object in tqdm(dataset.get_measure_all()):
        point = dataset.get_point_by_id(id_point=measure_object.idPoint)
        key = (point.id_train, measure_object.rangeType,
               measure_object.units, measure_object.param1, measure_object.param2)

        try:
            if measure_object.alarmLevel3:
                dd[key][0][0].add(measure_object.alarmLevel3)
            if measure_object.alarmLevel4:
                dd[key][0][1].add(measure_object.alarmLevel4)

            dd[key][1].append(measure_object.idMeasure)
        except (KeyError, TypeError):
            dd[key] = [[set(), set()], []]

            if measure_object.alarmLevel3:
                dd[key][0][0].add(measure_object.alarmLevel3)
            if measure_object.alarmLevel4:
                dd[key][0][1].add(measure_object.alarmLevel4)

            dd[key][1].append(measure_object.idMeasure)
        except Exception as exp:
            print(exp)

    for val in tqdm(dd.values()):
        alarms = val[0]
        if 0 in alarms[0]:
            alarms[0].remove(0)
        if 0 in alarms[1]:
            alarms[1].remove(0)

        alarm3 = None
        alarm4 = None

        if len(alarms[0]):
            alarm3 = min(alarms[0])
        if len(alarms[1]):
            alarm4 = min(alarms[1])
        if alarm3 and alarm4:
            if alarm3 > alarm4:
                alarm3, alarm4 = alarm4, alarm3

        alarms = [alarm3, alarm4]

        for id_mesaure in val[1]:
            if alarms[0] or alarms[1]:
                item = dataset.get_measure_by_id(id_measure=id_mesaure)
                if not item.alarmLevel3:
                    item.alarmLevel3 = alarms[0]
                if not item.alarmLevel4:
                    item.alarmLevel4 = alarms[1]
                dataset.update(item)


def big_refactor():
    """
    {id_train: [[ValueObject, ValueObject, ...], [ValueObject, ValueObject, ...], [ValueObject, ValueObject, ...] ...]}
    :return:
    """
    dataset = DataBase()

    dictionary = {}

    class ValueObject:
        def __init__(self, id_train, id_point, direction, unit, id_measure,
                     date, value, alarm3, alarm4, param1, param2):
            self.id_train = id_train
            self.id_point = id_point
            self.direction = direction
            self.unit = unit
            self.id_measure = id_measure
            self.date = date
            self.value = value
            self.alarm3 = alarm3
            self.alarm4 = alarm4
            self.param1 = param1
            self.param2 = param2

        def asdict(self):
            return {attr: self.__getattribute__(attr) for attr in self.__dict__}

    def first_iteration():
        """
        Загрузка всех элементов в словарь,
        ключ - id_train, значение - набор необходимых характеристик.
        """
        nonlocal dictionary

        print("\nЗагрузка данных в словарь")
        dictionary = {}
        for train_object in tqdm(dataset.get_train_all()):
            """
            Проход по всем значениям путем train -> point -> measure -> value. 
            Составление массивов, состоящих их value, для каждого ключа id_train в dictionary
            """
            # gf += 1
            # if gf == 100:
            #     break
            if train_object is None:
                print("Странно, но train_object is None")
                continue
            key = train_object.id_train
            dictionary[key] = {}
            for point_object in dataset.get_point_by_train(id_train=key):
                if point_object is None:
                    print("Странно, но point_object is None")
                    continue
                id_point = point_object.id_point
                dictionary[key][id_point] = []
                for measure_object in dataset.get_measure_by_point(id_point=id_point):
                    if measure_object is None:
                        print("Странно, но measure_object is None")
                        continue
                    id_measure = measure_object.id_measure

                    if measure_object.units not in [0, 1, 2]:
                        print(f"Странно, units = {measure_object.units}!")
                        print(f"id_train = {key}\nidPoint = {id_point}\nidMeasure = {id_measure}")
                        continue

                    for data_object in dataset.get_data_by_measure(id_measure=id_measure):
                        value_ = ValueObject(id_train=key, id_point=id_point, direction=point_object.direction,
                                             id_measure=id_measure, unit=measure_object.units,
                                             date=data_object.measure_date, value=data_object.value1,
                                             alarm3=measure_object.alarm_level3, alarm4=measure_object.alarm_level4,
                                             param1=measure_object.param1, param2=measure_object.param2)
                        dictionary[key][id_point].append(value_)
                dictionary[key][id_point].sort(key=lambda x: x.date)

    def second_iteration():
        nonlocal dictionary

        print("\nРазделение батчей по alarm3 и alarm4")
        for id_train in tqdm(dictionary):
            """
            Проход по полученным массивам в поисках превышения alarm4 (поломка)
            и в поиске сильного разрыва по дням.
            Если находим, то разделяем массив. Тем самым получается, что по 
            ключу id_train лежит массив из массивов.
            """
            for id_point in dictionary[id_train]:
                def split():
                    nonlocal arr, i, new_arr
                    new_arr.append(arr[: i + 1])
                    arr = arr[i + 1:]
                    i = 0

                arr = dictionary[id_train][id_point]
                new_arr = []
                i = 0
                while i < len(arr):
                    value_object: ValueObject = arr[i]
                    if i < len(arr) - 1:
                        next_object: ValueObject = arr[i + 1]

                        # Если происходит превышение по алармам4, то делаем сплит
                        if value_object.alarm4 and next_object.alarm4 \
                                and (
                                value_object.value > value_object.alarm4 or next_object.value > next_object.alarm4):
                            split()

                        # Если происходит превышение по алармам3, то делаем сплит
                        elif value_object.alarm3 and next_object.alarm3 \
                                and (
                                value_object.value > value_object.alarm3 or next_object.value > next_object.alarm3):
                            split()
                        # elif not 0 <= (next_object.date - value_object.date).days <= 5:
                        else:
                            i += 1
                    else:
                        break

                new_arr.append(arr)

                dictionary[id_train][id_point] = new_arr

    def third_iteration():
        """
        Интерполяция по каждому батчу
        """
        import matplotlib.pyplot as plt

        nonlocal dictionary

        print("\nДобавление сплайнов")
        direct_unit = [(i, l) for i in range(1, 4) for l in range(0, 3)]
        for id_train in tqdm(dictionary):
            for id_point in dictionary[id_train]:
                for k in range(len(dictionary[id_train][id_point])):
                    batch = dictionary[id_train][id_point][k]
                    if len(batch) < 4:
                        continue

                    for direction, unit in direct_unit:
                        same_batch = [value_object for value_object in batch
                                      if value_object.direction == direction and value_object.unit == unit]
                        if not same_batch:
                            continue

                        params = set((val.param1, val.param2) for val in same_batch)
                        for params_ in params:
                            bbatch = [val for val in same_batch
                                      if val.param1 == params_[0] and val.param2 == params_[1]]
                            date_arr = [time.mktime(value_object.date.timetuple()) for value_object in bbatch]
                            value_arr = [value_object.value for value_object in bbatch]

                            if len(date_arr) < 4:
                                continue

                            new_batches = []
                            step_unix = 60 * 60 * 15  # 15 часов
                            for i in range(1, len(bbatch)):
                                value_object_old: ValueObject = bbatch[i - 1]
                                value_object_next: ValueObject = bbatch[i]
                                new_batches.append(value_object_old)
                                t: timedelta = value_object_next.date - value_object_old.date
                                if 0 < t.days < 5:
                                    first_date_unix = time.mktime(value_object_old.date.timetuple())
                                    last_date_unix = time.mktime(value_object_next.date.timetuple())
                                    for l in range(int(first_date_unix + step_unix), int(last_date_unix), step_unix):
                                        new_value = copy(value_object_old)
                                        new_value.date = datetime.utcfromtimestamp(l)
                                        val = inter(x=date_arr, y=value_arr, x_val=l)
                                        if type(val) is not float:
                                            val = float(val)
                                            a = min(value_object_old.value, value_object_next.value)
                                            b = max(value_object_old.value, value_object_next.value)
                                            if not a <= val <= b:
                                                print("Опа, ошибка, хз почему")
                                                n = input()
                                        new_value.value = val
                                        new_batches.append(new_value)

                            new_batches.append(batch[-1])
                            batch.extend(new_batches)

                    batch.sort(key=lambda x: x.date)

        print("\nСохранение элементов в БД")
        o = 1
        id_trains = list(dictionary.keys())
        id_trains.sort()

        for id_train in tqdm(id_trains):
            r = []
            id_points = list(dictionary[id_train].keys())
            for id_point in id_points:
                for arr in dictionary[id_train][id_point]:
                    for value in arr:
                        r.append(value.asdict())

            r.sort(key=lambda x: x["date"])
            for i in range(1, len(r)):
                r[i - 1]["arr_idx"] = o
                # Если происходит превышение по алармам4, то делаем сплит
                if (r[i - 1]["alarm4"] and r[i]["alarm4"] and
                        (r[i - 1]["value"] > r[i - 1]["alarm4"] or r[i]["value"] > r[i]["alarm4"])):
                    o += 1

                r[i - 1] = EchkinaReadyTable(**r[i - 1])

            if r:
                r[-1]["arr_idx"] = o
                r[-1] = EchkinaReadyTable(**r[-1])

            o += 1
            r.sort(key=lambda x: (x.id_train, x.date))
            for obj in r:
                dataset.save(obj)

    first_iteration()
    second_iteration()
    third_iteration()


def inter(x, y, x_val):
    linear = interpolate.interp1d(x, y, kind="linear")
    return linear(x_val)


def ddd():
    x = [1637278672.0, 1637367828.0, 1637656361.0, 1637656366.0]
    val = [10.79915, 9.019785, 13.73455, 13.63186]
    target = 1637421828.0
    splain = interpolate.splrep(x, val, s=1)
    print(interpolate.splev(target, splain))
    # Полиномиальная интерполяция 351.10036413727255
    poly = interpolate.KroghInterpolator(x, val)
    print(poly(target))
    # "Кусочно-линейная интерполяция" 9.90217041172067
    linear = interpolate.interp1d(x, val, kind="linear")
    print(linear(target))
    # "Интерполяция кубическим сплайном" 351.1003641372645
    cubic = interpolate.interp1d(x, val, kind="cubic")
    print(cubic(target))



if __name__ == "__main__":
    big_refactor()
    # ddd()
