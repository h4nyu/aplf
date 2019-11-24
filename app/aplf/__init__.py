import os
from logging import getLogger, StreamHandler, Formatter, INFO
logger = getLogger()
logger.setLevel(INFO)
stream_handler = StreamHandler()
stream_handler.setLevel(INFO)
handler_format = Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
stream_handler.setFormatter(handler_format)
logger.addHandler(stream_handler)

config = {
    "TENSORBORAD_LOG_DIR": os.environ.get('TENSORBORAD_LOG_DIR', '/store/log')
}
