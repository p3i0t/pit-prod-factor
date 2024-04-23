import os
import pytest

from pit.configs import get_training_config, get_inference_config, ProdsAvailable

@pytest.mark.parametrize(
    'prod,milestone,expected',
    [
        ('0930', '2023-09-16', {'d_in': 278, 'd_out': 3, 'model': 'GPT_small'}),
        ('0930_1h', '2023-09-16', {'d_in': 278, 'd_out': 1, 'model': 'GPT_small'}),
        ('1030', '2023-09-16', {'d_in': 278, 'd_out': 3, 'model': 'GPT_small'}),
        ('1030_1h', '2023-09-16', {'d_in': 278, 'd_out': 1, 'model': 'GPT_small'}),
        ('1300', '2023-09-16', {'d_in': 278, 'd_out': 3, 'model': 'GPT_small'}),
        ('1300_1h', '2023-09-16', {'d_in': 278, 'd_out': 1, 'model': 'GPT_small'}),
        ('1400', '2023-09-16', {'d_in': 278, 'd_out': 3, 'model': 'GPT_small'}),
        ('1400_1h', '2023-09-16', {'d_in': 278, 'd_out': 1, 'model': 'GPT_small'}),
    ]
)
def test_training_configs(prod: ProdsAvailable, milestone: str, expected: dict):
    data_dir = os.getenv('DATASET_DIR')
    if data_dir is None:
        data_dir = "dataset"
        os.environ["DATASET_DIR"] = data_dir
    from pathlib import Path
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    args = get_training_config(prod, milestone)
    assert args.d_in == expected['d_in']
    assert args.d_out == expected['d_out']
    assert args.model == expected['model']
    
    
@pytest.mark.parametrize(
    'prod,expected',
    [
        ('0930', {'d_in': 278, 'd_out': 3, 'model': 'GPT_small', 'n_latest': 3}),
        ('0930_1h', {'d_in': 278, 'd_out': 1, 'model': 'GPT_small', 'n_latest': 3}),
        ('1030', {'d_in': 278, 'd_out': 3, 'model': 'GPT_small', 'n_latest': 3}),
        ('1030_1h', {'d_in': 278, 'd_out': 1, 'model': 'GPT_small', 'n_latest': 3}),
        ('1300', {'d_in': 278, 'd_out': 3, 'model': 'GPT_small', 'n_latest': 3}),
        ('1300_1h', {'d_in': 278, 'd_out': 1, 'model': 'GPT_small', 'n_latest': 3}),
        ('1400', {'d_in': 278, 'd_out': 3, 'model': 'GPT_small', 'n_latest': 3}),
        ('1400_1h', {'d_in': 278, 'd_out': 1, 'model': 'GPT_small', 'n_latest': 3}),
    ]
)
def test_inference_configs(prod: ProdsAvailable, expected: dict):
    data_dir = os.getenv('DATASET_DIR')
    if data_dir is None:
        data_dir = "dataset"
        os.environ["DATASET_DIR"] = data_dir
    from pathlib import Path
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    args = get_inference_config(prod)
    assert args.d_in == expected['d_in']
    assert args.d_out == expected['d_out']
    assert args.model == expected['model']
    assert args.prod == prod
    assert args.n_latest == expected['n_latest']
    
    