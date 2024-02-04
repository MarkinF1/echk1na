import logging


class CustomFormatter(logging.Formatter):

    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    def __init__(self, fmt):
        super().__init__()
        self.datefmt = '%d-%m-%Y %H:%M:%S'
        self.fmt = fmt
        self.FORMATS = {
            logging.DEBUG: self.grey + self.fmt + self.reset,
            logging.INFO: self.grey + self.fmt + self.reset,
            logging.WARNING: self.yellow + self.fmt + self.reset,
            logging.ERROR: self.red + self.fmt + self.reset,
            logging.CRITICAL: self.bold_red + self.fmt + self.reset
        }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


logging_lvl = logging.DEBUG
format = ("[%(asctime)s] [%(levelname)s] " +
          ("[(%(filename)s:%(lineno)d)]" if logging_lvl == logging.DEBUG else "") + "\n%(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging_lvl)
handlers = (logging.FileHandler('output.log'), logging.StreamHandler())
for hand in handlers:
    hand.setLevel(logging_lvl)
    hand.setFormatter(CustomFormatter(format))
    logger.addHandler(hand)

