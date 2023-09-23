from typing import List

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from create_db import EchkinaData, EchkinaTrain, EchkinaPoint


class Data:
    def __init__(self, data_object: EchkinaData):
        self.db_object = data_object

    def __str__(self):
        return f"id={self.id_} date={self.date} value={self.val}"

    @property
    def id_(self):
        return self.db_object.id

    @property
    def idMeasure(self):
        return self.db_object.idMeasure

    @property
    def date(self):
        return self.db_object.date

    @property
    def val(self):
        return self.db_object.value1


class MyDataset:
    engine = create_engine('postgresql://postgres:1@localhost:5432/echkina')

    def session(self):
        return Session(self.engine)

    # ----- Train -------
    def get_train_by_id(self, id_train: int | None) -> EchkinaTrain | None:
        if id_train is None:
            return

        with self.session() as session:
            return session.query(EchkinaTrain).filter(EchkinaTrain.idTrain == id_train).first()

    def get_train_by_name(self, name: str | None) -> List[EchkinaTrain]:
        if name is None:
            return []

        with self.session() as session:
            return session.query(EchkinaTrain).filter(EchkinaTrain.name == name).all()

    def remove_train_by_id(self, id_train: int | None):
        if id_train is None:
            return

        with self.session() as session:
            session.delete(EchkinaTrain(idTrain=id_train))

    def remove_train_by_name(self, name: str | None) -> None:
        if name is None:
            return

        with self.session() as session:
            session.delete(EchkinaTrain(name=name))

    # ------ Point -------
    def get_point_by_id(self, id_point: int | None) -> EchkinaPoint:
        if id_point is None:
            return

        with self.session() as session:
            return session.query(EchkinaPoint).filter(EchkinaPoint.idPoint == id_point).first()

    # ----- Data -------
    def get_data_all(self):
        with self.session() as session:
            return session.query(EchkinaData).all()

    def get_data_by_id(self, id_data: int | None) -> EchkinaData | None:
        if id_data is None:
            return

        with self.session() as session:
            return session.query(EchkinaData).filter(EchkinaData.id == id_data).first()

    def get_data_by_measure(self, id_measure: int | None) -> List[EchkinaData]:
        if id_measure is None:
            return []

        with self.session() as session:
            return session.query(EchkinaData).filter(EchkinaData.idMeasure == id_measure).all()

    def remove_data_by_id(self, id_data: int | None) -> None:
        if id_data is None:
            return

        with self.session() as session:
            item = session.get(EchkinaData, id_data)
            session.delete(item)
            session.commit()


if __name__ == "__main__":
    dataset = MyDataset()

    dataset.remove_data_by_id(1)
