import os
from collections import namedtuple
from logging import getLogger
from typing import Union, Optional

logger = getLogger(__name__)


class YamlCreator(object):
    def __init__(self):
        self._line = "---\n"

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

    def save(self, path: str, filename: str) -> None:
        try:
            with open(os.path.join(path, filename), "w+") as file:
                file.write(self._line)
        except Exception:
            logger.exception("%s Не удалось создать конфиг файл." % self._logger_abbreviation)
            exit(-1)
