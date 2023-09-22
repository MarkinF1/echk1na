import csv
import os


class Date:
    def __init__(self, date: str):
        self.is_correct = False

        self.year = None
        self.month = None
        self.day = None
        self.hours = None
        self.minutes = None
        self.seconds = None

        self.__make_date(date)

    def __str__(self):
        return "%s:%s:%s %s-%s-%s" % \
            (self.hours, self.minutes, self.seconds,
             self.day, self.month, self.year)

    def __make_date(self, date: str):
        date = date.split('-')

        if len(date) < 3:
            return
        self.year = int(date[0])
        self.month = int(date[1])
        date = date[2].split(' ')

        if len(date) < 2:
            return
        self.day = int(date[0])
        date = date[1].split(':')

        if len(date) < 3:
            return
        self.hours = int(date[0])
        self.minutes = int(date[1])
        self.seconds = int(date[2].split('.')[0])

        self.is_correct = True


class Data:
    def __init__(self, id_, date, val):
        self._id = int(id_)
        self._date = Date(date=date)
        self._val = float(val.replace(',', '.'))

    def __str__(self):
        return f"id={self.id_} date={self.date} value={self.val}"

    @property
    def id_(self):
        return self._id

    @property
    def date(self):
        return self._date

    @property
    def val(self):
        return self._val


class MyDataset:
    def __init__(self, dataset_path, data_file_path, info_file):
        self._dataset_path = dataset_path
        self._data_file = data_file_path
        self._info_file = info_file

        self._data_list = []

        self._train_dict = {}
        self._points_dict = {}
        self._measures_dict = {}

        self.__init_data_file()
        self.__init_info_file()

    def __init_data_file(self):
        with open(os.path.join(self._dataset_path, self._data_file)) as csv_file:
            reader = csv.reader(csv_file, delimiter=';')
            i = 0
            for row in reader:
                if i == 0:
                    i += 1
                    continue
                # if i == 100:
                #     break
                # i += 1
                self._data_list.append(Data(id_=row[0], date=row[1], val=row[2]))
        print(*self._data_list, sep='\n')
        print(len(self._data_list))

    def __init_info_file(self):
        pass


def main():
    dataset_path = "E:\datasets\echkina"
    data_file = "AllPumps_Data.csv"
    info_file = "Pumps_Struct_All.xlsx"

    dataset = MyDataset(dataset_path=dataset_path,
                        data_file_path=data_file,
                        info_file=info_file)


if __name__ == "__main__":
    main()
