import os
from omegaconf import OmegaConf

PIT_DIR = os.path.expanduser(os.getenv('PIT_DIR', '~/.pit'))
CONFIG_PATH = os.path.join(PIT_DIR, 'config.yml')
    
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
            'pit_dir': PIT_DIR,
            "tcalendar_path": '${pit_dir}/tcalendar.csv',
            'raw_dir': '${pit_dir}/raw',
            # all the raw data items
            'raw': {
                'bar_1m': {
                    'dir': '${raw.dir}/bar_1m',
                },
                'univ': {
                    'dir': '${raw.dir}/univ',
                },
                'barra': {
                    'dir': '${raw.dir}/barra',
                },
                'ohlcv_1m': {
                    'dir': '${raw.dir}/ohlcv_1m',
                },
                'return': {
                    'dir': '${raw.dir}/return',
                },
                'lag_return': {
                    'dir': '${raw.dir}/lag_return',
                },
            },
            'derived_dir': '${pit_dir}/derived',
            # all the derived data items
            'derived': {
                'bar_10m': {
                    'dir': '${derived_dir}/bar_10m',
                },
            },
            'dataset_dir': '${pit_dir}/dataset',
            'dataset': {
                '10m_v2': {
                    'dir': '${dataset_dir}/10m_v2',
                }
            },
            'save_dir': '${pit_dir}/runs',
            'infer_dir': '${pit_dir}/infer',
        }
        
        cfg = OmegaConf.create(default_config)
        OmegaConf.save(cfg, CONFIG_PATH)
    return PIT_DIR


def read_config():
    cfg = OmegaConf.load(CONFIG_PATH)
    OmegaConf.resolve(cfg)
    return cfg