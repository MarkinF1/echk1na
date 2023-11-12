import time
from copy import copy

from scipy import interpolate
import os
import re
from collections import namedtuple
from datetime import datetime, timedelta
from random import shuffle
from typing import Optional, List

from tqdm import tqdm
from yaml import load, SafeLoader

from main import MyDataset
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
    file.to_csv(f"{CONFIG.dataset_path}/{CONFIG.info_measures_path}", index=False,  sep=';')


def push_data_to_db():
    from create_db import EchkinaTrain, EchkinaPoint, EchkinaMeasure, EchkinaData, engine
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
        columns = ["idTrain", "name", "description"]
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
        columns = ["idTrain", "idPoint", "name", "description", "direction", "controllParametrType"]
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
    from main import MyDataset

    dataset = MyDataset()
    for data_object in dataset.get_data_all():
        d = data_object.date
        if d.year < 2020:
            dataset.remove_data_by_id(data_object.id)


def delete_old_measures_from_db():
    from main import MyDataset
    from tqdm import tqdm

    dataset = MyDataset()
    for measure in tqdm(dataset.get_measure_all()):
        if not dataset.get_data_by_measure(measure.idMeasure):
            dataset.remove_measure_by_id(id_measure=measure.idMeasure)


def add_type_of_class_data():
    from main import MyDataset
    from tqdm import tqdm

    dataset = MyDataset()
    dd = {}
    for measure_object in tqdm(dataset.get_measure_all()):
        point = dataset.get_point_by_id(id_point=measure_object.idPoint)
        key = (point.idTrain, measure_object.rangeType,
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
    {idTrain: [[ValueObject, ValueObject, ...], [ValueObject, ValueObject, ...], [ValueObject, ValueObject, ...] ...]}
    :return:
    """
    dataset = MyDataset()

    dictionary = {}

    class ValueObject:
        def __init__(self, id_train, id_point, direction, unit, id_measure, date, value, alarm3, alarm4):
            self.id_train = id_train
            self.id_point = id_point
            self.direction = direction
            self.unit = unit
            self.id_measure = id_measure
            self.date = date
            self.value = value
            self.alarm3 = alarm3
            self.alarm4 = alarm4

        def asdict(self):
            return {attr: self.__getattribute__(attr) for attr in self.__dict__}

    class SmartContainer:
        def __init__(self, direction: int, unit: int):
            self.direction = direction
            self.unit = unit

            self.batches: List[List[ValueObject]] = []
            self.current_add_batch: List[ValueObject] = []

        def add(self, elem: ValueObject) -> None:
            if self.current_add_batch:
                delay = elem.date - self.current_add_batch[-1].date
                if not 0 <= delay.days <= 5:
                    self.flush()

            self.current_add_batch.append(elem)

        def flush(self) -> None:
            self.batches.append(self.current_add_batch)
            self.current_add_batch = []

        def interpolate(self, num_days: int = 5, num_hours: int = 15):
            """
            Выполняет flush, а посел аппроксимацию где разница дней меньше num_days.
            :param num_days: максимальное количество дней для аппроксимации.
            :return:
            """
            self.flush()
            for l in range(len(self.batches)):
                batch = self.batches[l]
                if len(batch) < 4:
                    continue

                batch.sort(key=lambda x: x.date)
                date_arr = [time.mktime(value_object.date.timetuple()) for value_object in batch]
                value_arr = [value_object.value for value_object in batch]

                splain = interpolate.splrep(date_arr, value_arr)

                new_batches = []
                step_unix = 60 * 60 * num_hours
                for i in range(1, len(batch)):
                    value_object_old = batch[i - 1]
                    value_object_next = batch[i]
                    new_batches.append(value_object_old)
                    t: timedelta = value_object_next.date - value_object_old.date
                    if 0 < t.days < num_days:
                        first_date_unix = time.mktime(value_object_old.date.timetuple())
                        last_date_unix = time.mktime(value_object_next.date.timetuple())
                        for i in range(int(first_date_unix + step_unix), int(last_date_unix), step_unix):
                            new_value = copy(value_object_old)
                            new_value.date = datetime.utcfromtimestamp(i)
                            new_value.value = interpolate.splev(i, splain)
                            new_batches.append(new_value)
                new_batches.append(batch[-1])
                self.batches[l] = new_batches

        def delete_small_batches(self, num: int):
            for i in range(len(self.batches) - 1, -1, -1):
                if len(self.batches[i]) < num:
                    self.batches.pop(i)

        def __eq__(self, other):
            return other.direction == self.direction and other.unit == self.unit

    class ContainerCtrl:
        def __init__(self):
            self.containers = [SmartContainer(direction=i, unit=j) for i in range(1, 4) for j in range(0, 3)]

        def add_trains(self, dictionary: dict) -> None:
            for arrays in dictionary.values():
                for arr in arrays:
                    for value_object in arr:
                        value_object: ValueObject
                        container = self.get_container(direction=value_object.direction, unit=value_object.unit)
                        if container:
                            container.add(value_object)
                    self.flush_all()

        def get_dict(self):
            dictionary = {}
            for container in self.containers:
                for batch in container.batches:
                    if batch:
                        try:
                            dictionary[batch[0].id_train].append(batch)
                        except KeyError:
                            dictionary[batch[0].id_train] = [batch]
            return dictionary

        def get_container(self, direction: int, unit: int) -> Optional[SmartContainer]:
            for container in self.containers:
                if container.direction == direction and container.unit == unit:
                    return container
            else:
                print(f"ContainerCtrl: я не нашел контейнер с direction = {direction}, unit = {unit}")
                return None

        def flush_all(self) -> None:
            for container in self.containers:
                container.flush()

        def interpolate(self, num_days: int = 5) -> None:
            """
            Выполняет кубическую интерполяцию для всех SmartControllers.
            :param num_days: количество дней разрыва, если больше, то не выолняется
            """
            for container in self.containers:
                container.interpolate(num_days=num_days)

        def save(self) -> None:
            pass

        def delete_small_batches(self, num_elements: int):
            for container in self.containers:
                container.delete_small_batches(num_elements)

    def first_iteration():
        nonlocal dictionary
        # if os.path.exists("get_all_value.yml"):
        #     with open("get_all_value.yml", "r") as file:
        #         dictionary = load(file, Loader=SafeLoader)["dictionary"]
        # else:
        dictionary = {}
        for train_object in tqdm(dataset.get_train_all()[:5]):
            """
            Проход по всем значениям путем train -> point -> measure -> value. 
            Составление массивов, состоящих их value, для каждого ключа id_train в dictionary
            """

            if train_object is None:
                print("Странно, но train_object is None")
                continue
            key = train_object.idTrain
            dictionary[key] = []
            for point_object in dataset.get_point_by_train(id_train=key):
                if point_object is None:
                    print("Странно, но point_object is None")
                    continue
                id_point = point_object.idPoint
                for measure_object in dataset.get_measure_by_point(id_point=id_point):
                    if measure_object is None:
                        print("Странно, но measure_object is None")
                        continue
                    id_measure = measure_object.idMeasure

                    if measure_object.units not in [0, 1, 2]:
                        print(f"Странно, units = {measure_object.units}!")
                        print(f"idTrain = {key}\nidPoint = {id_point}\nidMeasure = {id_measure}")
                        continue

                    for data_object in dataset.get_data_by_measure(id_measure=id_measure):
                        value_ = ValueObject(id_train=key, id_point=id_point, direction=point_object.direction,
                                             id_measure=id_measure, unit=measure_object.units,
                                             date=data_object.date, value=data_object.value1,
                                             alarm3=measure_object.alarmLevel3, alarm4=measure_object.alarmLevel4)
                        dictionary[key].append(value_)
            dictionary[key].sort(key=lambda x: x.date)
            # else:
            #     yaml_file = YamlCreator()
            #
            #     for lst in dictionary.values():
            #         for i in range(len(lst)):
            #             lst[i] = lst[i].asdict()
            #
            #     yaml_file.add_parameter(name="dictionary", value=dictionary)
            #     yaml_file.save(path="./", filename="get_all_value.yml")
            #     del yaml_file
        #
        # for lst in dictionary.values():
        #     for i in range(len(lst)):
        #         lst[i] = value(**lst[i])

    def second_iteration():
        nonlocal dictionary
        # if os.path.exists("cut_all_alarms.yml"):
        #     with open("cut_all_alarms.yml", "r") as file:
        #         dictionary = load(file, Loader=SafeLoader)["dictionary"]
        # else:
        for id_train in tqdm(dictionary):
            """
            Проход по полученным массивам в поисках превышения alarm4 (поломка)
            и в поиске сильного разрыва по дням.
            Если находим, то разделяем массив. Тем самым получается, что по 
            ключу id_train лежит массив из массивов.
            """
            def split():
                nonlocal arr, i
                new_arr.append(arr[: i + 1])
                arr = arr[i + 1:]
                i = 0

            dictionary[id_train].sort(key=lambda x: x.date)
            arr = dictionary[id_train]
            new_arr = []
            i = 0
            while i < len(arr):
                value_object: ValueObject = arr[i]
                if i < len(arr) - 1:
                    next_object: ValueObject = arr[i + 1]
                    if not 0 <= (next_object.date - value_object.date).days <= 5:
                        if value_object.alarm4 and next_object.alarm4 \
                            and value_object.value > value_object.alarm4 \
                            and next_object.value > next_object.alarm4:
                            split()
                            continue
                        if value_object.alarm3 and next_object.alarm3 \
                            and value_object.value > value_object.alarm3 \
                            and next_object.value > next_object.alarm3:
                            split()


                if value_object.alarm4 is None or value_object.value < value_object.alarm4:
                    i += 1
                    continue

            new_arr.append(arr)

            dictionary[id_train] = new_arr
            # else:
            #     yaml_file = YamlCreator()
            #
            #     for lst in dictionary.values():
            #         for i in range(len(lst)):
            #             lst[i] = lst[i].asdict()
            #
            #     yaml_file.add_parameter(name="dictionary", value=dictionary)
            #     yaml_file.save(path="./", filename="cut_all_alarms.yml")
            #     del yaml_file
        #
        # for lst in dictionary.values():
        #     for i in range(len(lst)):
        #         lst[i] = value(**lst[i])

    def third_iteration():
        """
        Интерполяция по каждому батчу
        :return:
        """
        nonlocal dictionary
        controller = ContainerCtrl()
        controller.add_trains(dictionary=dictionary)
        controller.interpolate()
        controller.delete_small_batches(num_elements=30)
        dictionary = controller.get_dict()
        for value in dictionary.values():
            value.sort(key=lambda x: x[0].date)
        cd = ContainerCtrl()

    first_iteration()
    second_iteration()
    third_iteration()


if __name__ == "__main__":
    # xlsx2csv()
    # asyncio.run(main())
    # delete_columns_data(old_path=os.path.join(CONFIG.dataset_path, CONFIG.data_path),
    #                     new_path=os.path.join(CONFIG.dataset_path, CONFIG.new_data_path))

    # delete_duplicate_rows(tables=[(os.path.join(CONFIG.dataset_path, CONFIG.info_train_path),
    #                               os.path.join(CONFIG.dataset_path, CONFIG.new_info_train_path)),
    #                               (os.path.join(CONFIG.dataset_path, CONFIG.info_points_path),
    #                               os.path.join(CONFIG.dataset_path, CONFIG.new_info_points_path)),
    #                               (os.path.join(CONFIG.dataset_path, CONFIG.info_measures_path),
    #                               os.path.join(CONFIG.dataset_path, CONFIG.new_info_measures_path))])

    # push_data_to_db()
    # delete_old_data_from_db()
    # delete_old_measures_from_db()
    # test()
    # trainer = Trainer()
    # loader()
    # add_type_of_class_data()
    # add_alarm_to_data()
    big_refactor()
