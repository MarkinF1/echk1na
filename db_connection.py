import datetime
from typing import List, Generator, Optional

from sqlalchemy import create_engine, and_
from sqlalchemy.orm import Session

from db_classes import EchkinaData, EchkinaTrain, EchkinaPoint, EchkinaMeasure, EchkinaTmpTable, EchkinaReadyTable, \
    EchkinaReadyTableCrop


class DataBase:
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

    def get_train_by_id(self, id_train: Optional[int]) -> Optional[EchkinaTrain]:
        if id_train is None:
            return

        with self.session() as session:
            return session.query(EchkinaTrain).get(id_train)

    def get_train_by_name(self, name: Optional[str]) -> List[EchkinaTrain]:
        if name is None:
            return []

        with self.session() as session:
            return session.query(EchkinaTrain).filter(EchkinaTrain.name == name).all()

    def remove_train_by_id(self, id_train: Optional[int]):
        if id_train is None:
            return

        with self.session() as session:
            session.delete(EchkinaTrain(id_train=id_train))

    def remove_train_by_name(self, name: Optional[str]) -> None:
        if name is None:
            return

        with self.session() as session:
            session.delete(EchkinaTrain(train_name=name))

    # ------ Point -------
    def get_point_by_id(self, id_point: Optional[int]) -> Optional[EchkinaPoint]:
        if id_point is None:
            return

        with self.session() as session:
            return session.query(EchkinaPoint).get(id_point)

    def get_point_by_train(self, id_train: Optional[int]) -> List[EchkinaPoint]:
        if id_train is None:
            return []

        with self.session() as session:
            return session.query(EchkinaPoint).filter(EchkinaPoint.id_train == id_train).all()

    # ---- Measure ------
    def get_measure_by_id(self, id_measure: int) -> Optional[EchkinaMeasure]:
        if id_measure is None:
            return

        with self.session() as session:
            return session.query(EchkinaMeasure).get(id_measure)

    def get_measure_by_point(self, id_point: Optional[int]) -> List[EchkinaMeasure]:
        if id_point is None:
            return []

        with self.session() as session:
            return session.query(EchkinaMeasure).filter(EchkinaMeasure.id_point == id_point).all()

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

    def get_data_by_id(self, id_data: Optional[int]) -> Optional[EchkinaData]:
        if id_data is None:
            return

        with self.session() as session:
            return session.query(EchkinaData).get(id_data)

    def get_data_by_measure(self, id_measure: Optional[int]) -> List[EchkinaData]:
        if id_measure is None:
            return []

        with self.session() as session:
            return session.query(EchkinaData).filter(EchkinaData.id_measure == id_measure).all()

    def remove_data_by_id(self, id_data: Optional[int]) -> None:
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
    def get_ready_data_by_id(self, id_data: Optional[int]) -> Optional[EchkinaReadyTable]:
        """
        Возвращает EchkinaReadyTable по его айдишнику.
        :param id_data:
        :return:
        """
        if id_data is None:
            return

        with self.session() as session:
            return session.query(EchkinaReadyTable).get(id_data)

    def get_ready_data_all_id(self) -> List[int]:
        """
        Возвращает все айдишники, которые есть в таблице EchkinaReadyTable.
        :return:
        """
        with self.session() as session:
            output = [int(row[0]) for row in session.query(EchkinaReadyTable.id).all()]
            output.sort()
            return output

    def get_ready_data_by_train(self, id_train: Optional[int]) -> List[EchkinaReadyTable]:
        """
        Возвращает все элементы EchkinaReadyTable с id_train.
        :param id_train:
        :return:
        """
        if id_train is None:
            return []

        with self.session() as session:
            elements = session.query(EchkinaReadyTable).filter(EchkinaReadyTable.id_train == id_train).all()
            elements.sort(key=lambda x: (x.arr_idx, x.date))
            return elements

    def get_ready_data_all_train(self) -> List[int]:
        """
        Возвращает все отсортированные id_train.
        :return:
        """
        with self.session as session:
            output = set(int(row[0]) for row in session.query(EchkinaReadyTable.id_train).all())
            return sorted(list(output))

    def get_ready_data_all_train_date_crop(self) -> List[EchkinaReadyTableCrop]:
        """
        Возвращает все элементы таблицы EchkinaReadyTable, но обрезанный,
        только id, id_train, date, arr_idx
        :return:
        """
        with self.session() as session:
            output = [EchkinaReadyTableCrop(**{"id_": row[0], "id_train": row[1], "date": row[2], "arr_idx": row[3]})
                      for row in session.query(EchkinaReadyTable.id,
                                               EchkinaReadyTable.id_train,
                                               EchkinaReadyTable.date,
                                               EchkinaReadyTable.arr_idx).all()]

            return output

    def get_ready_data_by_unit_direction_crop(self, unit: int, direction: int) -> List[EchkinaReadyTableCrop]:
        """
        Возвращает обрезанные элементы таблицы EchkinaReadyTable по unit и direction
        :param unit:
        :param direction:
        :return:
        """
        with self.session() as session:
            output = [EchkinaReadyTableCrop(row[0], row[1], row[2], row[3])
                      for row in session.
                      query(EchkinaReadyTable.id,
                            EchkinaReadyTable.id_train,
                            EchkinaReadyTable.date,
                            EchkinaReadyTable.arr_idx).
                      order_by(EchkinaReadyTable.arr_idx,
                               EchkinaReadyTable.date).
                      filter(EchkinaReadyTable.unit == unit,
                             EchkinaReadyTable.direction == direction).
                      all()
                      ]

            output.sort(key=lambda x: (x.arr_idx, x.date))
            return output

    def get_ready_data_special(self, id_train: int,
                               max_date: datetime.datetime, arr_idx: int) -> List[EchkinaReadyTable]:
        with self.session() as session:
            condition = and_(EchkinaReadyTable.id_train == id_train,
                             EchkinaReadyTable.date <= max_date,
                             EchkinaReadyTable.arr_idx == arr_idx)
            result = session.query(EchkinaReadyTable).filter(condition).all()
            return result


def main():
    dataset = DataBase()
    dict_of_models = {}
    # for data_object in dataset.get_data_all():
        # try:
        #     dict_of_models[data_object.id_measure] += 1
        # except KeyError:
        #     dict_of_models[data_object. easure] = 1

    print(f"len: {len(dict_of_models)}\ndict_of_models: {dict_of_models}")


if __name__ == "__main__":
    main()
