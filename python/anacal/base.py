import logging


def setup_custom_logger(verbose: bool = False):
    """Set up a custom logger with the specified format and level.

    Args:
    verbose (bool): whether verbose

    Returns:
    logger : logger of anacal
    """
    logger = logging.getLogger("anacal")
    if verbose:
        level = logging.INFO
    else:
        level = logging.WARNING

    logger.setLevel(level)

    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="%(asctime)s %(message)s",
            datefmt="%Y/%m/%d %H:%M:%S --- ",
        )
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        handler.setLevel(level)
        logger.addHandler(handler)
    return logger


class AnacalBase(object):
    def __init__(self, verbose, logger=None):
        if logger is None:
            self.logger = setup_custom_logger(verbose)
        else:
            self.logger = logger
