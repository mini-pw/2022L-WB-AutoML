import logging
import logging.config

logging_level = logging.DEBUG
logger = logging.getLogger(__name__)
logger.setLevel(logging_level)
import os
print(os.getcwd())
with open('log.txt', 'w') as file:
    pass
logger_handler = logging.FileHandler(filename = 'log.txt')
logger_handler.setLevel(logging_level)
logger_formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
logger_handler.setFormatter(logger_formatter)
logger.addHandler(logger_handler)
logger.info('Completed configuring logger()!')


from .OurFlaml import OurFlaml

__version__ = '0.1.0'
