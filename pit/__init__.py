"""Pit: A Python package for printing money.
"""
from pit.prod import get_training_config, get_inference_config, ProdsAvailable, list_prods, get_bars # noqa
from pit.train import TrainPipeline  # noqa
from pit.inference import InferencePipeline  # noqa

import datetime
__version__ = datetime.datetime.now().strftime("%Y.%m.%d")
