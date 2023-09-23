from sqlalchemy.orm import declarative_base, relation, relationship
from sqlalchemy import create_engine, Column, ForeignKey, Integer, DateTime, Float, String, Boolean

Base = declarative_base()


class EchkinaTrain(Base):
    __tablename__ = "train"

    idTrain = Column(Integer, primary_key=True)
    name = Column(String)
    description = Column(String)


class EchkinaPoint(Base):
    __tablename__ = "points"

    idPoint = Column(Integer, primary_key=True)
    idTrain = Column(Integer, ForeignKey("train.idTrain"))
    name = Column(String)
    description = Column(String)
    direction = Column(Integer)
    controllParametrType = Column(Integer)
    # train = relation("Train")


class EchkinaMeasure(Base):
    __tablename__ = "measures"

    idMeasure = Column(Integer, primary_key=True)
    idPoint = Column(Integer, ForeignKey("points.idPoint"))
    name = Column(String)
    description = Column(String)
    type_ = Column(Integer)
    rangeType = Column(Integer)
    units = Column(Integer)
    param1 = Column(Integer)
    param2 = Column(Integer)
    param3 = Column(Integer)
    alarmType = Column(Boolean)
    alarmLevel2 = Column(Float)
    alarmLevel3 = Column(Float)
    alarmLevel4 = Column(Float)
    # point = relation("Point")


class EchkinaData(Base):
    __tablename__ = "data"

    id = Column(Integer, primary_key=True)
    idMeasure = Column(Integer)
    date = Column(DateTime)
    value1 = Column(Float)
    # measure = relation("Measure")
