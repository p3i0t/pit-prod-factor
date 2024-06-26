import os
from omegaconf import OmegaConf

PIT_DIR = os.path.expanduser(os.getenv("PIT_DIR", "~/.pit"))
CONFIG_PATH = os.path.join(PIT_DIR, "config.yml")

# Register a new resolver named "mkdir"
OmegaConf.register_new_resolver("mkdir", lambda x: os.makedirs(x, exist_ok=True) or x)

# In the example's configuration setup, you didn't directly see a resolver being called like ${mkdir:...} within the YAML file.
# Instead, the directory creation was invoked programmatically during runtime:

# Trigger directory creation with side effects
# OmegaConf.resolve(config)

# This invocation of OmegaConf.resolve(config) goes through the configuration object and resolves any interpolations or resolver functions.
# If there were expressions like ${mkdir:/path/to/dir} directly in the configuration, they would get evaluated at this point—causing the directories to be created.


def init_config() -> str:
    os.makedirs(PIT_DIR, exist_ok=True)
    if not os.path.exists(CONFIG_PATH):
        default_config = {
            "pit_dir": PIT_DIR,
            "tcalendar_path": "${pit_dir}/tcalendar.csv",
            # all the raw data items
            "raw": {
                "dir": "${pit_dir}/raw", 
                "dataitems": ["bar_1m", "univ", "barra", "ohlcv_1m", "return", "lag_return", "tick"],
            },
            # all the derived data items
            "derived": {
                "dir": "${pit_dir}/derived",
                "dataitems": ["bar_10m"]
            },
            "dataset": {
                "dir": "${pit_dir}/dataset",
                "dataitems": ["10m_v2"],
            },
            "save_dir": "${pit_dir}/runs",
            "infer_dir": "${pit_dir}/infer",
        }

        cfg = OmegaConf.create(default_config)
        OmegaConf.save(cfg, CONFIG_PATH)
        
        # create all directories
        OmegaConf.resolve(cfg)
        from pathlib import Path
        Path(cfg.raw.dir).mkdir(parents=True, exist_ok=True)
        for item in cfg.raw.dataitems:
            Path(f"{cfg.raw.dir}").joinpath(item).mkdir(parents=True, exist_ok=True)
        Path(cfg.derived.dir).mkdir(parents=True, exist_ok=True)
        for item in cfg.derived.dataitems:
            Path(f"{cfg.derived.dir}").joinpath(item).mkdir(parents=True, exist_ok=True)
        Path(cfg.dataset.dir).mkdir(parents=True, exist_ok=True)
        for item in cfg.dataset.dataitems:
            Path(f"{cfg.dataset.dir}").joinpath(item).mkdir(parents=True, exist_ok=True)
        Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)
        Path(cfg.infer_dir).mkdir(parents=True, exist_ok=True)

    return PIT_DIR


def read_config():
    cfg = OmegaConf.load(CONFIG_PATH)
    OmegaConf.resolve(cfg)
    return cfg
