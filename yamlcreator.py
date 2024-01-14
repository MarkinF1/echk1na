import os
from logger import logger
from typing import Union, Optional


class YamlCreator(object):
    def __init__(self):
        self._line = "---\n"
        self._values = {}  # словарь вида {name: [value, description]}

    def __new__(cls, *args, **kwargs):
        cls._logger_abbreviation = "%s:" % cls.__name__
        return super(YamlCreator, cls).__new__(cls)

    def add_parameter(self,
                      name: str,
                      value: Union[str, dict, set, list, float, bool, int, None],
                      description: Optional[str] = None,
                      tab: str = '') -> None:
        def recursive_get_values(tab_curr: str,
                                 key: Union[str, tuple],
                                 value_: Union[str, dict, set, list, float, bool, int, None]) -> str:
            line = "%s%s" % (tab + tab_curr, key)
            if type(value_) in [list, set]:
                if line[-2:] != '- ':
                    line += '\n'
                for i, val in enumerate(value_):
                    line += recursive_get_values('' if i == 0 and line[-2:] == '- ' else tab_curr + const_tab,
                                                 '- ', val)
            elif type(value_) is dict:
                line += '\n'
                for key, val in value_.items():
                    if isinstance(key, tuple):
                        line += "%s# %s\n" % (tab_curr + const_tab, key[1])
                        key = key[0]

                    line += recursive_get_values(tab_curr + const_tab, "%s: " % key, val)
            elif value_ is None:
                line += "\n"
            else:
                line += "%s\n" % value_
            return line

        const_tab = " "
        self._line += '\n'
        if description:
            self._line += "# %s\n" % description

        self._line += recursive_get_values('', "%s: " % name, value)

        return

    def add_description(self, description: str, tab: str = ''):
        self._line += "%s# %s\n" % (tab, description)

    # TODO: в будущем переименовать, херня какая-то
    def add_line(self):
        self._line += '\n'

    def remove_parameter(self, key: str):
        try:
            delete_description = self._values[key][1] is not None
            self._values.pop(key)

        except KeyError:
            logger.info("%s Параметр %s не найден в self._values для удаления" % (self._logger_abbreviation, key))
            return -1

        lines = self._line.split('\n')
        i = len(lines) - 1
        while i > -1:
            if lines[i].startswith(key):
                # удаление основной переменной
                lines.pop(i)

                # удаление если у переменной есть вложенность (массивы, словари)
                while i < len(lines) and lines[i].startswith(' '):
                    lines.pop(i)

                if not delete_description:
                    break

                i -= 1
                if not len(lines) or i == -1:
                    break

                # удаление если у переменной есть описание
                if 0 <= i < len(lines) and delete_description and lines[i].startswith('#'):
                    lines.pop(i)
                break

            i -= 1
        else:
            logger.info("%s Параметр %s не найден в self._line для удаления" % (self._logger_abbreviation, key))
            return -1

        self._line = '\n'.join(lines)
        return i

    def save(self, path: str, filename: str) -> None:
        try:
            with open(os.path.join(path, filename), "w+") as file:
                file.write(self._line)
        except (OSError, FileExistsError, ValueError):
            logger.exception("%s Не удалось создать/открыть конфиг файл." % self._logger_abbreviation)
            exit(1)
