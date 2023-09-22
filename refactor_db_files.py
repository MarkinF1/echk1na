from collections import namedtuple
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


def main():
    pass


if __name__ == "__main__":
    main()
    # delete_columns_data(old_path=CONFIG.data_path, new_path=CONFIG.new_data_path)
    # delete_duplicate_rows(tables=[(CONFIG.info_train_path, CONFIG.new_info_train_path),
    #                               (CONFIG.info_points_path, CONFIG.new_info_points_path),
    #                               (CONFIG.info_measures_path, CONFIG.new_info_measures_path)])
