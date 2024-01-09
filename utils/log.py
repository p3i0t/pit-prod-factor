import logging
from pathlib import Path


def get_logger(name: str, parent: str = None, level=logging.INFO, formatter='%(asctime)s: %(levelname)s - %(message)s'):
    logger = logging.getLogger(name=name)
    logger.setLevel(level=level)
    # don't forget to set the logger level to be no higher than handler levels, otherwise you won't see anything from your logger
    formatter = logging.Formatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level=level)
    stream_handler.setFormatter(formatter)
    if parent:
        file_path = Path(parent) / f"{name}.log"
    else:
        file_path = f"{name}.log"
    # print(f"log file to: {file_path}")
    file_handler = logging.FileHandler(str(file_path), mode='a')
    file_handler.setLevel(level=level)
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.propagate = False
    return logger