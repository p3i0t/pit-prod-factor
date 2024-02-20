"""Pit: A Python package for printing money.
"""
import os

# default env variables that to be set in use.
os.environ["DATASET_DIR"] = "dataset"  # directory to processed dataset.
os.environ["CALENDAR_PATH"] = "calendar.pkl"  # path to trading calendar.
os.environ['SAVE_DIR'] = 'runs'  # directory to save checkpoints.

from pit.configs import get_training_config, get_inference_config, ProdsAvailable, list_prods, get_bars # noqa
from pit.train import TrainPipeline  # noqa
from pit.inference import InferencePipeline  # noqa

__version__ = "0.6.66"
