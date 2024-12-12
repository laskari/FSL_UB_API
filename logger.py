# logger.py



import logging

import os

from logging.handlers import TimedRotatingFileHandler

import sys



def setup_logger(name):

    formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    directory = r"/Data/FSl_ML/FSL_codebase/api/UB/logging_dir"

    os.makedirs(directory, exist_ok=True)

    error_file = os.path.join(directory, "ub_logs.txt")

    handler = TimedRotatingFileHandler(error_file, when="midnight", interval=1, backupCount=15)

    handler.suffix = "%Y-%m-%d"

    handler.setFormatter(formatter)

    screen_handler = logging.StreamHandler(stream=sys.stdout)

    screen_handler.setFormatter(formatter)

    logger = logging.getLogger(name)

    logger.setLevel(logging.DEBUG)

    logger.addHandler(handler)

    logger.addHandler(screen_handler)

    return logger



ub_logger = setup_logger('UB')

