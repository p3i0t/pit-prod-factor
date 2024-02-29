import os
import click
from omegaconf import OmegaConf
# from typing import Literal


# @click.command()
# @click.option('--source', default='bopu', help='source of calendar data')
# def update_calendar(source: Literal['akshare', 'bopu'] = 'bopu'):
#     import os
#     import pickle
#     script_directory = os.path.dirname(os.path.abspath(__file__))
#     if source == "akshare":
#         save_path = os.path.join(script_directory, 'ak_calendar.pkl')
#         import akshare as ak
#         df = ak.tool_trade_date_hist_sina()
#         trade_dates = [d.strftime('%Y-%m-%d') for d in sorted(df['trade_date']) if d.strftime('%Y-%m-%d') > '2017-01-01']
#         pickle.dump(trade_dates, open(save_path, 'wb'))
#     else:
#         try:
#             import genutils as gu  # noqa
#         except ImportError:
#             gu = None
#         trade_dates = gu.get_trade_dates()
#         save_path = os.path.join(script_directory, 'bopu_calendar.pkl')
#     return trade_dates

@click.command()
@click.option('--prod', default='1030', help='product to train')
@click.option('--milestone', default='today', help='milestone date of the model to train.')
def train_single(prod, milestone):
    """cli command to train single model of given prod and milestone.
    """
    from pit import get_training_config, TrainPipeline, list_prods
    # pit_dir = os.path.join(
    #     os.getenv("PIT_HOME", os.path.expanduser("~")), ".pit")
    # conf = OmegaConf.load(open(f"{pit_dir}/config.yml"))
    # config = OmegaConf.to_container(conf)
    # os.environ['DATASET_DIR'] = config['DATASET_DIR']
    # os.environ['CALENDAR_PATH'] = config['CALENDAR_PATH']
    # os.environ['SAVE_DIR'] = config['SAVE_DIR']
    
    
    valid_prods = list_prods()
    if prod not in valid_prods:
        raise ValueError(f"prod {prod} not in {valid_prods}")
    args = get_training_config(prod=prod, milestone=milestone)
    pipe = TrainPipeline(args)
    pipe.run()
    
    
@click.command()
@click.option('--prod', default='1030', help='product to infer')
@click.option('--infer_date', default='today', help='the date of data used for inference.')
def inference(prod, infer_date):
    """cli command to train single model of given prod and milestone.
    
    For prod used at 0930 of next trading day, the date in the result is the next trading day after infer_date.
    """
    import sys
    from pit import get_inference_config, list_prods
    from pit.inference import infer, InferenceDataSource
    from pit.utils import normalize_date
    import polars as pl
    from datetime import timedelta
    from pathlib import Path
    import genutils as gu
    
    infer_date = normalize_date(infer_date).strftime('%Y-%m-%d')
    valid_prods = list_prods()
    if prod not in valid_prods:
        raise ValueError(f"prod {prod} not in {valid_prods}")
    args = get_inference_config(prod=prod)
    
    if len(gu.tcalendar.get(infer_date, infer_date)) == 0:
        print(f"generate date {infer_date} is not a trading date !!!")
        sys.exit(0)
    
    o = infer(args=args, infer_date=infer_date, mode=InferenceDataSource.online, debug=False)
    assert isinstance(o, pl.DataFrame)
    if prod in ['0930', '0930_1h']:
        next_date = gu.tcalendar.adjust(infer_date, 1)
        # replace date with next_date
        o = o.with_columns(pl.lit(next_date).cast(pl.Date).alias('date'))

    slot = args.y_slots
    assert isinstance(slot, str)
    o = o.with_columns(
        pl.col('date')
        .cast(pl.Datetime(time_unit='ns'))
        .add(timedelta(hours=int(slot[:2]), minutes=int(slot[2:])))
        .alias('time')
    )
    
    pit_dir = os.path.join(os.getenv("PIT_HOME", os.path.expanduser("~")), ".pit")
    infer_dir = Path(OmegaConf.load(open(f"{pit_dir}/config.yml")).INFER_DIR)
    tgt_dir = infer_dir.joinpath(prod)
    tgt_dir.mkdir(parents=True, exist_ok=True)
    
    alpha_col = [col for col in o.columns if col.endswith('2D')]
    assert len(alpha_col) == 1
    alpha_col = alpha_col[0]
    
    alpha = o.select(['date', 'time', 'symbol', alpha_col]).rename(mapping={alpha_col: 'pit'})
    alpha.write_parquet(tgt_dir.joinpath(f"{infer_date}.parq"))
    
    

@click.group()
def pit():
    # """Manage my hahaha package."""
    click.echo("Alpha Signals Generator of Pit.")
    # pit_dir is the directory to store the trading calendar, config file, results of this package.
    pit_dir = os.path.join(
        os.getenv("PIT_HOME", os.path.expanduser("~")), ".pit")

    if not os.path.exists(pit_dir):
        os.makedirs(pit_dir)
        click.echo(f"Created directory: {pit_dir}")

    else:
        pass
        # click.echo(f"Directory already exists: {pit_dir}")
    
    if not os.path.exists(f"{pit_dir}/config.yml"):
        click.echo(f"Config file missing! generating default config file at {pit_dir}/config.yml")
        
        default_config = {
            "DATASET_DIR": "/data2/private/wangxin/dataset/10m_v2",
            "CALENDAR_PATH": f"{pit_dir}/calendar.pkl",
            "SAVE_DIR": f"{pit_dir}/runs",
            "INFER_DIR": f"{pit_dir}/inference",
        }
        
        cfg = OmegaConf.create(default_config)
        OmegaConf.save(cfg, f"{pit_dir}/config.yml")
        
    # ToDO: update calendar.pkl
    # if not os.path.exists(f"{pit_dir}/calendar.pkl"):
    #     click.echo(f"Generating default calendar file at {pit_dir}/calendar.pkl")
    #     from pit.utils import update_calendar
    #     trade_dates = update_calendar()
    #     pickle.dump(trade_dates, open(f"{pit_dir}/calendar.pkl", 'wb'))


# pit.add_command(data)
pit.add_command(train_single)
pit.add_command(inference)

if __name__ == "__main__":
    pit()