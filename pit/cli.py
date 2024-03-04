import os
import click
from omegaconf import OmegaConf
from typing import Literal, Callable
from loguru import logger

from pit import list_prods, get_bars, get_training_config, get_inference_config, TrainPipeline
from pit.utils import any2ymd


# self-defined click ParamType by overriding the convert() method 
# to convert the value (as str) to the desired type (date str)
class ExtendedDate(click.ParamType):
    name = 'date'
    def convert(self, value, param, ctx):
        try:
            return any2ymd(str(value))
        except ValueError:
            self.fail(f"{value} is not a valid date or 'today'.", param, ctx)

# Instantiate the custom type to use with the click option
DateType = ExtendedDate()

@click.command()
@click.option(
    "--path", '-p', "dir",
    default="/data2/private/wangxin/raw2",
    type=click.Path(),
    help="path directory to store the downloaded data, will be created if doesn't exist.",
)

@click.option(
    "--begin",
    default="2017-01-01",
    type=DateType,
    help="begin date, e.g. '20210101', '2021-01-01', 'today'.",
)
@click.option(
    "--end",
    default="today",
    type=DateType,
    help="end date, e.g. '20231001', '2023-10-01', or `today`.",
)
@click.option(
    "--item",
    default="univ",
    type=click.Choice(["bar_1m", "univ", "return", "lag_return"]),
    help="item name to download, one of `bar_1m`, `univ`, `return`, `lag_return`.",
)
@click.option("--n_jobs", default=10, type=int, 
              help="number of ray parallel jobs at most, only used for bar_1min.")
def download(
    dir: str,
    begin: str,
    end: str,
    item: Literal["bar_1m", "univ", "return", "lag_return"],
    n_jobs: int,
):
    """Manage data for pit."""
    from pathlib import Path
    from datetime import datetime, timedelta
    
    import ray
    import polars as pl
    from pit.utils import any2date
    from pit.data import BopuDataReader

    _dir = Path(dir)
    _dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Download {item} to directory={_dir}")
    if datetime.now().strftime("%H%M") < "1530":
        hard_end = datetime.now() - timedelta(hours=24)
    else:
        hard_end = datetime.now()
    _end = min(any2date(end), any2date(hard_end))

    try:
        import genutils as gu
    except ImportError:
        raise ImportError("Please install genutils first.")
    import os

    trading_dates = sorted(gu.tcalendar.get(begin=begin, end=_end))
    trading_dates = [d.strftime("%Y-%m-%d") for d in trading_dates]

    item_dir = _dir.joinpath(item)
    item_dir.mkdir(parents=True, exist_ok=True)

    existing_dates = [d.split(".")[0] for d in os.listdir(item_dir)]
    if item == "return" or item == "lag_return":
        existing_dates = sorted(existing_dates)[:-6]
    trading_dates = sorted(set(trading_dates) - set(existing_dates))

    if len(trading_dates) == 0:
        click.echo(f"{item} is up to date {datetime.now().date()}")
        return
    click.echo(
        f"Download {item} to directory={item_dir}, {len(trading_dates)} tasks to be done."
    )

    reader = BopuDataReader()
    from functools import partial

    fn_dict: dict[str, Callable[..., pl.DataFrame]] = {
        "bar_1m": partial(reader.fetch_1min_bars, cols=get_bars("v3")),
        "univ": reader.fetch_univs,
        # 'lag': bopu_reader.fetch_lag,
        "return": partial(reader.fetch_returns, n_list=[1, 2, 5]),
        "lag_return": partial(reader.fetch_lag_returns, n_list=[1, 2, 5]),
    }

    ray.init(num_cpus=n_jobs, ignore_reinit_error=True, include_dashboard=False)
    @ray.remote(max_calls=2)
    def wrap_with_save(fn: Callable[..., pl.DataFrame], begin: str, end: str, /) -> None:
        df: pl.DataFrame = fn(begin=begin, end=end)
        
        res_list = []
        for (d, ), _df in df.partition_by(["date"], as_dict=True).items():
            _df.write_parquet(f"{item_dir}/{d:%Y-%m-%d}.parq")
            res_list.append(d)

    if item == 'bar_1m':
        task_ids = []
        n_task_finished = 0
        for exp_id, d in enumerate(trading_dates, 1):
            task_id = wrap_with_save.options(
                name="x",
                num_cpus=1,
            ).remote(fn=fn_dict[item], begin=d, end=d)
            task_ids.append(task_id)

            if len(task_ids) >= n_jobs:
                dones, task_ids = ray.wait(task_ids, num_returns=1)
                ray.get(dones)
                n_task_finished += 1
                logger.info(f"{n_task_finished} tasks finished.")
        ray.get(task_ids)
    else:
        task_id = wrap_with_save.options(
            name='x'
            ).remote(fn=fn_dict[item], begin=trading_dates[0], end=trading_dates[-1])
        ray.get(task_id)
    click.echo(f"{item} done.")


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
@click.option('--prod', '-p', 
              default=click.Choice(list_prods()),
              help='product to infer')
@click.option('--milestone', default='today', help='milestone date of the model to train.')
def train_single(prod, milestone):
    """cli command to train single model of given prod and milestone.
    """
    args = get_training_config(prod=prod, milestone=milestone)
    pipe = TrainPipeline(args)
    pipe.run()
 
@click.command()
@click.option('--prod', '-p', 
              default=click.Choice(list_prods()),
              help='product to infer')
@click.option('--mode', '-m', default='train', 
              type=click.Choice(['train', 'inference']),
              help='train or inference args to show.')
def show(prod, mode):
    """cli command to list training or inference args.
    """
    if mode == 'train':
        args = get_training_config(prod=prod)
    elif mode == 'inference':
        args = get_inference_config(prod=prod)
    else:
        raise ValueError(f"mode {mode} not in ['train', 'inference']")
    print(args)

    
@click.command()
@click.option('--prod', '-p', 
              default=click.Choice(list_prods()),
              help='product to infer')
@click.option('--date', '-d', default='today', help='the date of data used for inference.')
def inference(prod, date):
    """cli command to train single model of given prod and milestone.
    
    For prod used at 0930 of next trading day, the date in the result is the next trading day after infer_date.
    """
    import sys
    from pit.inference import infer, InferenceDataSource
    from pit.utils import any2ymd 
    import polars as pl
    from datetime import timedelta
    from pathlib import Path
    import genutils as gu
    
    infer_date: str = any2ymd(date)
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
    
    alpha = o.select(['date', 'time', 'symbol', args.tgt_column]).rename(mapping={args.tgt_column: 'pit'})
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
pit.add_command(show)
pit.add_command(download)

if __name__ == "__main__":
    pit()