from typing import List, Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from create_db import EchkinaData, EchkinaTrain, EchkinaPoint, EchkinaMeasure, EchkinaTmpTable, EchkinaReadyTable


class MyDataset:
    engine = create_engine('postgresql://postgres:postgres@localhost:5432/echkina')

    def session(self):
        return Session(self.engine)

    def update(self, db_object):
        with self.session() as session:
            session.merge(db_object)
            session.commit()

    def save(self, db_object):
        if db_object is None:
            return

        with self.session() as session:
            session.add(db_object)
            session.commit()

    # ----- Train -------
    def get_train_all(self):
        with self.session() as session:
            return session.query(EchkinaTrain).all()

    def get_train_by_id(self, id_train: int | None) -> EchkinaTrain | None:
        if id_train is None:
            return

        with self.session() as session:
            return session.query(EchkinaTrain).get(id_train)

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
    def get_point_by_id(self, id_point: int | None) -> EchkinaPoint | None:
        if id_point is None:
            return

        with self.session() as session:
            return session.query(EchkinaPoint).get(id_point)

    def get_point_by_train(self, id_train: int | None) -> List[EchkinaPoint]:
        if id_train is None:
            return []

        with self.session() as session:
            return session.query(EchkinaPoint).filter(EchkinaPoint.idTrain == id_train).all()

    # ---- Measure ------
    def get_measure_by_id(self, id_measure: int) -> EchkinaMeasure | None:
        if id_measure is None:
            return

        with self.session() as session:
            return session.query(EchkinaMeasure).get(id_measure)

    def get_measure_by_point(self, id_point: int | None) -> List[EchkinaMeasure]:
        if id_point is None:
            return []

        with self.session() as session:
            return session.query(EchkinaMeasure).filter(EchkinaMeasure.idPoint == id_point).all()

    def get_measure_all(self) -> List[EchkinaMeasure]:
        with self.session() as session:
            return session.query(EchkinaMeasure).all()

    def remove_measure_by_id(self, id_measure: int):
        if id_measure is None:
            return

        with self.session() as session:
            item = session.get(EchkinaMeasure, id_measure)
            session.delete(item)
            session.commit()

    # ----- Data -------
    def get_data_all(self) -> Generator[None, EchkinaData, None]:
        with self.session() as session:
            count = session.query(EchkinaData).count()
            for id_ in range(767294, count + 767294):
                object = session.query(EchkinaData).get(id_)
                if object is not None:
                    yield object

    def get_data_by_id(self, id_data: int | None) -> EchkinaData | None:
        if id_data is None:
            return

        with self.session() as session:
            return session.query(EchkinaData).get(id_data)

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
            if item:
                session.delete(item)
                session.commit()

    def get_data_count(self):
        with self.session() as session:
            return session.query(EchkinaData).count()

    # ----- Доп функции ------
    def get_tmp_data_all(self):
        with self.session() as session:
            return session.query(EchkinaTmpTable).all()

    def get_tmp_by_id_measure(self, id_measure: int) -> List[EchkinaTmpTable]:
        with self.session() as session:
            return session.query(EchkinaTmpTable).filter(EchkinaTmpTable.idMeasure == id_measure).all()

    # ---- ReadyData таблица -----
    def get_ready_data_by_id(self, id_data: int | None) -> EchkinaReadyTable | None:
        if id_data is None:
            return

        with self.session() as session:
            return session.query(EchkinaReadyTable).get(id_data)


def main():
    dataset = MyDataset()
    dict_of_models = {}
    for data_object in dataset.get_data_all():
        try:
            dict_of_models[data_object.idMeasure] += 1
        except KeyError:
            dict_of_models[data_object.idMeasure] = 1

    print(f"len: {len(dict_of_models)}\ndict_of_models: {dict_of_models}")


if __name__ == "__main__":
    main()
