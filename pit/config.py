import os
from omegaconf import OmegaConf

PIT_DIR = os.path.expanduser(os.getenv('PIT_DIR', '~/.pit'))
CONFIG_PATH = os.path.join(PIT_DIR, 'config.yml')
    
def init_config() -> str:
    os.makedirs(PIT_DIR, exist_ok=True)
    
    if not os.path.exists(CONFIG_PATH):
        TCALENDAR_PATH = os.path.join(PIT_DIR, 'tcalendar.csv') # visible
        RAW_DIR = os.path.join(PIT_DIR, 'raw')
        DERIVED_DIR = os.path.join(PIT_DIR, 'derived')
        
        # raw_config = {
        #     'bar_1m': os.path.join(RAW_DIR, 'bar_1m'),
        #     'univ': os.path.join(RAW_DIR, 'univ'),
        #     'barra': os.path.join(RAW_DIR, 'barra'),
        #     'ohlcv_1m': os.path.join(RAW_DIR, 'ohlcv_1m'),
        #     'return': os.path.join(RAW_DIR, 'return'),
        #     'lag_return': os.path.join(RAW_DIR, 'lag_return'),
        # }
        # derived_config = {
        #     'bar_10m': os.path.join(DERIVED_DIR, 'bar_10m'),
        # }
        default_config = {
            'tcalendar_path': TCALENDAR_PATH,
            'raw_dir': RAW_DIR,
            # 'raw': raw_config,
            'derived_dir': DERIVED_DIR,
            # 'derived': derived_config,
        }
        
        cfg = OmegaConf.create(default_config)
        OmegaConf.save(cfg, CONFIG_PATH)
    return PIT_DIR

def read_config():
    return OmegaConf.load(CONFIG_PATH)