import datetime

from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, ForeignKey, Integer, DateTime, Float, String, Boolean, Date, TIMESTAMP

Base = declarative_base()

# ---------------------------
# Классы для обработки данных
class EchkinaTrain(Base):
    __tablename__ = "train_6"

    id_train = Column(Integer, primary_key=True)
    train_name = Column(String)
    description = Column(String)


class EchkinaPoint(Base):
    __tablename__ = "point_6"

    id_train = Column(Integer, ForeignKey("train.idTrain"))
    id_point = Column(Integer, primary_key=True)
    point_name = Column(String)
    description = Column(String)
    direction = Column(Integer)
    controlled_parameter_type = Column(Integer)
    # train = relation("Train")


class EchkinaMeasure(Base):
    __tablename__ = "measure_6"

    id_point = Column(Integer, ForeignKey("points.idPoint"))
    id_measure = Column(Integer, primary_key=True)
    measure_name = Column(String)
    description = Column(String)
    measure_type = Column(Integer)
    range_type = Column(Integer)
    units = Column(Integer)
    param1 = Column(Integer)
    param2 = Column(Integer)
    param3 = Column(Integer)
    alarm_type = Column(Boolean)
    alarm_level2 = Column(Float)
    alarm_level3 = Column(Float)
    alarm_level4 = Column(Float)
    id_train = Column(Integer)
    # point = relation("Point")


class EchkinaData(Base):
    __tablename__ = "data_6"

    id_measure = Column(Integer, primary_key=True)
    measure_date = Column(DateTime, primary_key=True)
    value1 = Column(Float)
    value2 = Column(Float)


class EchkinaTmpTable(Base):
    __tablename__ = "tmp_one_more"

    day = Column(Date)
    id = Column(Integer, primary_key=True)
    idMeasure = Column(Integer)
    date = Column(TIMESTAMP)
    value1 = Column(Float)
    alarm = Column(Integer)


# ---------------------------
# Классы для обучения и тестирования модели
class EchkinaReadyTable(Base):
    __tablename__   = "ready_data"

    id              = Column(Integer, primary_key=True, server_default='uuid_generate_v4()')
    id_train        = Column(Integer, nullable=False)
    id_point        = Column(Integer, nullable=False)
    id_measure      = Column(Integer, nullable=False)
    direction       = Column(Integer)
    unit            = Column(Integer)
    date            = Column(TIMESTAMP, nullable=False)
    value           = Column(Float, nullable=False)
    alarm3          = Column(Float)
    alarm4          = Column(Float)
    param1          = Column(Float)
    param2          = Column(Float)
    arr_idx         = Column(Integer, nullable=False)


class EchkinaReadyTableCrop:
    id_: int
    id_train: int
    date: datetime.datetime
    arr_idx: int

    def __init__(self, id_: int, id_train: int, date: datetime.datetime, arr_idx: int):
        self.id_ = id_
        self.id_train = id_train
        self.date = date
        self.arr_idx = arr_idx
