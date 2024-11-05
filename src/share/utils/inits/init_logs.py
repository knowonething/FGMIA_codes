import os.path
import socket
import sys

import diffusers
import torch
import transformers
from loguru import logger as log


def initLogs(config):
    log.remove()
    log_format = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{" \
                 "name}</cyan>:<cyan>{function}</cyan> <cyan>{file}</cyan>:<cyan>{line}</cyan> - <level>{" \
                 "message}</level> "
    log.add(sys.stdout, format=log_format)
    filename = os.path.join(config.init_outputDir, "logs", "%s/INFO_{time}.log" % socket.gethostname())
    log.add(filename, format=log_format, encoding="utf-8")
    log.info("pytorch:{}", torch.__version__)
    log.info("transformers:{}", transformers.__version__)
    log.info("cuda:{}", torch.cuda.is_available())
    log.info("cuda version:{}", torch.version.cuda)
    log.info("cudnn version:{}", torch.backends.cudnn.version())
    log.info("diffusers version:{}", diffusers.__version__)
