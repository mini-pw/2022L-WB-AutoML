import logging
import logging.config
import sys


#ustaw na DEBUG żeby pokazywały się informacje z preprocessingu
logging_level = logging.INFO
logger = logging.getLogger()
fhandler = logging.FileHandler(filename=".log.txt", mode = 'w')
logger.addHandler(fhandler)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(lineno)s - %(message)s')
fhandler.setFormatter(formatter)
consoleHandler = logging.StreamHandler(sys.stdout)
consoleHandler.setFormatter(formatter)

