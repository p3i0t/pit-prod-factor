import os

from omegaconf import OmegaConf


def read_config():
  PIT_DIR = os.path.expanduser(os.getenv("PIT_DIR", "~/.pit"))
  default_config = {
    "pit_dir": PIT_DIR,
    # all the raw data items
    "raw": {
      "dir": "${pit_dir}/raw",
      "dataitems": ["bar_1m", "univ", "return", "lag_return"],
    },
    "dataset": {
      "dir": "${pit_dir}/dataset",
      "dataitems": ["10m_v2"],
    },
    "save_dir": "${pit_dir}/runs",
    "infer_dir": "${pit_dir}/infer",
  }

  cfg = OmegaConf.create(default_config)
  # OmegaConf.save(cfg, CONFIG_PATH)

  # create all directories
  OmegaConf.resolve(cfg)
  from pathlib import Path

  Path(cfg.raw.dir).mkdir(parents=True, exist_ok=True)
  for item in cfg.raw.dataitems:
    Path(f"{cfg.raw.dir}").joinpath(item).mkdir(parents=True, exist_ok=True)
  Path(cfg.dataset.dir).mkdir(parents=True, exist_ok=True)
  for item in cfg.dataset.dataitems:
    Path(f"{cfg.dataset.dir}").joinpath(item).mkdir(parents=True, exist_ok=True)
  Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)
  Path(cfg.infer_dir).mkdir(parents=True, exist_ok=True)

  return cfg
