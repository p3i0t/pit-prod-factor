"""Pit: A Python package for printing money.
"""
from pit.prod import get_training_config, get_inference_config, ProdsAvailable, list_prods, get_bars # noqa
from pit.train import TrainPipeline  # noqa
from pit.inference import InferencePipeline  # noqa

__version__ = "2024.07.16"
