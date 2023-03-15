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
                f' [D] loss: {record.D:>7.5f};' )
    
    logger.info(f' [G  ]:' + 
                f' Total loss: {record.Gtotal.G:>7.5f}'  +
                f' Adv loss: {record.Gtotal.GAdv:>7.5f}' +
                f' Pxl loss: {record.Gtotal.GPxl:>7.5f}' +
                f' Per loss: {record.Gtotal.GPer:>7.5f}' )
    
    names = ['g ', 'tl', 'tr', 'd ']
    for i in range(4):
        logger.info(f' [G{names[i]}]:' + 
                    f' Total loss: {record.Gparts[i].G:>7.5f}'  +
                    f' Adv loss: {record.Gparts[i].GAdv:>7.5f}' +
                    f' Pxl loss: {record.Gparts[i].GPxl:>7.5f}' +
                    f' Per loss: {record.Gparts[i].GPer:>7.5f}' +
                    ("\n" if i==3 else "") )


def write_loss(writer, record:LossRecord, tag1:str, step:int) -> None:
    writer.add_scalar(f'{tag1}_D/D_loss', record.D, step)
    
    writer.add_scalar(f'{tag1}_G/Total', record.Gtotal.G,    step)
    writer.add_scalar(f'{tag1}_G/Adv',   record.Gtotal.GAdv, step)
    writer.add_scalar(f'{tag1}_G/Pxl',   record.Gtotal.GPxl, step)
    writer.add_scalar(f'{tag1}_G/Per',   record.Gtotal.GPer, step)
    
    names = ['g ', 'tl', 'tr', 'd ']
    for i in range(4):
        writer.add_scalar(f'{tag1}_G_{names[i]}/Total', record.Gparts[i].G,    step)
        writer.add_scalar(f'{tag1}_G_{names[i]}/Adv',   record.Gparts[i].GAdv, step)
        writer.add_scalar(f'{tag1}_G_{names[i]}/Pxl',   record.Gparts[i].GPxl, step)
        writer.add_scalar(f'{tag1}_G_{names[i]}/Per',   record.Gparts[i].GPer, step)
