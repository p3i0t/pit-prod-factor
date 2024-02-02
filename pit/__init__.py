"""Pit: A Python package for printing money.
"""
import os
os.environ["DATASET_DIR"] = "dataset"

from pit.configs import get_training_config, get_inference_config, ProdsAvailable  # noqa
from pit.train import TrainPipeline  # noqa
from pit.inference import InferencePipeline  # noqa



__version__ = "0.6.6"
