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
                dd[key][0][0] = measure_object.alarmLevel3
            if measure_object.alarmLevel4:
                dd[key][0][1] = measure_object.alarmLevel4

            dd[key][1].append(measure_object.idMeasure)
        except KeyError:
            dd[key] = [[0, 0], []]

            if measure_object.alarmLevel3:
                dd[key][0][0] = measure_object.alarmLevel3
            if measure_object.alarmLevel4:
                dd[key][0][1] = measure_object.alarmLevel4

            dd[key][1].append(measure_object.idMeasure)
        except Exception as exp:
            print(exp)

    for val in tqdm(dd.values()):
        alarms = val[0]
        for idMesaure in val[1]:
            if alarms[0] or alarms[1]:
                item = dataset.get_measure_by_id(id_measure=idMesaure)
                item.alarmLevel3 = alarms[0]
                item.alarmLevel4 = alarms[1]
                dataset.update(item)

def loader(self):
    from main import MyDataset

    dataset = MyDataset()
    trains = dataset.get_train_by_id(9338)
    shuffle(trains)
    for train_object in trains:
        values = ()
        points = self.dataset.get_point_by_train(id_train=train_object.idTrain)
        for point_object in points:
            measures = self.dataset.get_measure_by_point(id_point=point_object.idPoint)
            for measure_object in measures:
                datas = self.dataset.get_data_by_measure(id_measure=measure_object.idMeasure)
                for data_object in datas:
                    values = values + data_object.value1


class Trainer:
    def __init__(self, is_small_predict=True):
        from main import MyDataset

        self.days = 3 if is_small_predict else 14
        self.count_values = 20 if is_small_predict else 40
        self.dataset = MyDataset

    def loader(self):
        trains = self.dataset.get_train_all()
        shuffle(trains)
        for train_object in trains:
            values = ()
            points = self.dataset.get_point_by_train(id_train=train_object.idTrain)
            for point_object in points:
                measures = self.dataset.get_measure_by_point(id_point=point_object.idPoint)
                for measure_object in measures:
                    datas = self.dataset.get_data_by_measure(id_measure=measure_object.idMeasure)
                    for data_object in datas:
                        values = values + data_object.value1
                        if len(values) == self.count_values:
                            yield values + train_object.idTrain

    def train(self):
        pass


def add_alarm_to_data():
    dataset = MyDataset()

    for data in tqdm(dataset.get_tmp_data_all()):
        measure = dataset.get_measure_by_id(id_measure=data.idMeasure)
        if measure is None:
            dataset.remove_data_by_id(id_data=data.id)
            continue

        if measure.alarmLevel3 is not None or measure.alarmLevel4 is not None:
            data.alarm = 0

            if measure.alarmLevel3 is not None and data.value1 > measure.alarmLevel3:
                data.alarm = 1

            if measure.alarmLevel4 is not None and data.value1 > measure.alarmLevel4:
                data.alarm = 2

            dataset.update(data)


def big_refactor():
    """
    {idTrain: [[ValueObject, ValueObject, ...], [ValueObject, ValueObject, ...], [ValueObject, ValueObject, ...] ...]}
    :return:
    """
    dataset = MyDataset()
    v = namedtuple('ValueObject',
                   ("id_point", "direction", "unit", "id_measure", "day",
                    "date", "value", "alarm3", "alarm4"))

    class value(v):
        def asdict(self):
            return self._asdict()

    if os.path.exists("get_all_value"):
        with open("get_all_value", "r") as file:
            dictionary = load(file, Loader=SafeLoader)
    else:
        dictionary = {}
        for train_object in tqdm(dataset.get_train_all()[:3]):
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

                    for tmp_object in dataset.get_tmp_by_id_measure(id_measure=id_measure):
                        value_ = value(id_point=id_point, direction=point_object.direction,
                                       id_measure=id_measure, unit=measure_object.units, day=tmp_object.day,
                                       date=tmp_object.date, value=tmp_object.value1,
                                       alarm3=measure_object.alarmLevel3, alarm4=measure_object.alarmLevel4)
                        dictionary[key].append(value_)
            dictionary[key].sort(key=lambda x: x[2])
        else:
            yaml_file = YamlCreator()

            yaml_file.add_parameter(name="dictionary", value=dictionary)
            yaml_file.save(path="./", filename="get_all_value.yml")
            del yaml_file

    if os.path.exists("cut_all_alarms"):
        with open("cut_all_alarms", "r") as file:
            dictionary = json.load(file)
    else:
        for id_train in tqdm(dictionary):
            """
            Проход по полученным массивам в поисках превышения alarm4 (поломка).
            Если находим, то разделяем массив. Тем самым получается, что по 
            ключу id_train лежит массив из массивов.
            """
            arr = dictionary[id_train]
            new_arr = []
            i = 0
            while i < len(arr):
                value_object: value = arr[i]
                if value_object.alarm4 is None or value_object.value < value_object.alarm4:
                    i += 1
                    continue

                new_arr.append(arr[: i + 1])
                arr = arr[i + 1:]
                i = 0
            new_arr.append(arr)
            dictionary[id_train] = new_arr
        else:
            with open("cut_all_alarms", "w+") as file:
                json.dump(dictionary, file)

    class SmartContainer:
        def __init__(self, direction: int, unit: int):
            self.direction = direction
            self.unit = unit

            self.batches: List[List[value]] = []
            self.current_add_batch: List[value] = []

        def add(self, elem: value) -> None:
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
                if not len(batch):
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
                            new_value.day = datetime(year=new_value.date.year,
                                                     month=new_value.date.month,
                                                     day=new_value.date.day)
                            new_value.value = interpolate.splev(i, splain)
                            new_batches.append(new_value)
                new_batches.append(batch[-1])
                self.batches[l] = new_batches

        def __eq__(self, other):
            return other.direction == self.direction and other.unit == self.unit

    class ContainerCtrl:
        def __init__(self):
            self.containers = [SmartContainer(direction=i, unit=j) for i in range(1, 4) for j in range(0, 3)]

        def add_trains(self, dictionary: dict) -> None:
            for arrays in dictionary.values():
                for arr in arrays:
                    for value_object in arr:
                        value_object: value
                        container = self.get_container(direction=value_object.direction, unit=value_object.unit)
                        if container:
                            container.add(value_object)
                    self.flush_all()

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

    if os.path.exists("get_controller"):
        with open("get_controller") as file:
            controller = json.load(file)
    else:
        controller = ContainerCtrl()
        controller.add_trains(dictionary=dictionary)
        controller.interpolate()
        with open("get_controller", "w+") as file:
            json.dump(controller, file)


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
