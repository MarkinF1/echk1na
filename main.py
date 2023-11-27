import sklearn
from db_connection import DataBase


class DataLoader:
    def __init__(self, days: int = 14):
        self.__days: int = days

        self.__is_train: bool = True
        self.__train_ids: list = []
        self.__test_ids: list = []
        self.__database = DataBase()

        self.__init()

    def __init(self) -> None:
        ids = self.__database.get_ready_data_all_id()
        self.__train_ids, self.__test_ids = sklearn.model_selection.train_test_split(ids, test_size=0.2)

    def
    def __getitem__(self, item):
        pass

    def train(self):
        self.is_train = True

    def eval(self):
        self.is_train = False

    