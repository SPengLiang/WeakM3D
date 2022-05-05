import os
import sys
import loguru
import time

def prepare_dirs(cfg):
    if not os.path.exists(cfg.LOG_DIR):
        os.makedirs(cfg.LOG_DIR)
    if not os.path.exists(cfg.CHECKPOINTS_DIR):
        os.makedirs(cfg.CHECKPOINTS_DIR)

def log_write(log, out):
    log.write(out+'\n')
    log.flush()
    print(out)

def init_logger(cfg, log_prefix="train"):
    def get_time_str(fmt="%Y%m%d_%H%M%S"):
        r"""get format time string"""
        return time.strftime(fmt, time.localtime())
    log_dir = os.path.join(cfg.LOG_DIR, cfg.EXP_NAME)
    # use clean format style in stdout but detail in log file
    clean_style = (
        "<green>{time:YYYY-MM-DD HH:mm}</green> |"
        "<level>{level: <6}</level> | <level>{message}</level>"
    )
    loguru.logger.remove()
    loguru.logger.add(sys.stdout, enqueue=True, format=clean_style)
    detail_style = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <6}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>"
        " - <level>{message}</level>"
    )
    logger_file = os.path.join(
        log_dir, f"{log_prefix}_{cfg.EXP_NAME}_{get_time_str()}.log"
    )
    loguru.logger.add(logger_file, enqueue=True, format=detail_style)

    loguru.logger.info(f"config: \n{cfg}")

