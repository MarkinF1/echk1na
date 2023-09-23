import asyncio
import os
import re
from collections import namedtuple
from datetime import datetime

from sqlalchemy import DateTime
from yaml import load, SafeLoader

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

async def main():
    from cmc_hw.data_creater.loader.loader import get_points_by_trains
    async for point in get_points_by_trains([12935]):
        print(point, end='\n')


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

    push_data_to_db()
