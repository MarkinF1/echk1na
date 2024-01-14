import logging


class CustomFormatter(logging.Formatter):

    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    def __init__(self):
        super().__init__()
        self.datefmt = '%d-%m-%Y %H:%M:%S'
        format = "[%(asctime)s] [%(levelname)s] [(%(filename)s:%(lineno)d)]\n%(message)s"
        self.fmt = format
        self.FORMATS = {
            logging.DEBUG: self.grey + format + self.reset,
            logging.INFO: self.grey + format + self.reset,
            logging.WARNING: self.yellow + format + self.reset,
            logging.ERROR: self.red + format + self.reset,
            logging.CRITICAL: self.bold_red + format + self.reset
        }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


logging_lvl = logging.DEBUG
logger = logging.getLogger(__name__)
logger.setLevel(logging_lvl)
handlers = (logging.FileHandler('output.log'), logging.StreamHandler())
for hand in handlers:
    hand.setLevel(logging_lvl)
    hand.setFormatter(CustomFormatter())
    logger.addHandler(hand)

