import time
import logging

from utils.LossRecord import LossRecord


def get_logger(logDir:str) -> logging.Logger:
    nowTime = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))
    logPath = f"{logDir}/{nowTime}.log"
    
    logger = logging.getLogger(logPath)
    logger.setLevel(logging.DEBUG)
    
    if logger.root.handlers:
        logger.root.handlers[0].setLevel(logging.WARNING)
    
    formatter = logging.Formatter('[%(asctime)s] %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
    
    fh = logging.FileHandler(logPath)
    sh = logging.StreamHandler()
    fh.setLevel(logging.INFO)
    sh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    sh.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(sh)
    
    return logger


def log_loss(logger:logging.Logger, record:LossRecord, epoch:int) -> None:
    logger.info(f'Epoch: {epoch:>3d};' +
                f' D loss: {record.D:>7.5f};'       +
                f' G loss: {record.G:>7.5f};'       +
                f' GAdv loss: {record.GAdv:>7.5f};' +
                f' GPxl loss: {record.GPxl:>7.5f};' +
                f' GPer loss: {record.GPer:>7.5f};' + '\n')


def write_loss(writer, record:LossRecord, tag2:str, step:int) -> None:
    writer.add_scalar(f'D_loss/{tag2}', record.D, step)
    writer.add_scalar(f'G_loss/{tag2}', record.G, step)
    writer.add_scalar(f'GAdv_loss/{tag2}', record.GAdv, step)
    writer.add_scalar(f'GPxl_loss/{tag2}', record.GPxl, step)
    writer.add_scalar(f'GPer_loss/{tag2}', record.GPer, step)
