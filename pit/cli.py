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
@click.option("--n_jobs", default=10, type=int, 
              help="number of ray parallel jobs.")
def download_1m(
    dir: str,
    begin: str,
    end: str,
    n_jobs: int,
):
    """Download 1min bars."""
    from pathlib import Path
    from datetime import datetime, timedelta
    
    import ray
    import polars as pl
    from pit.utils import any2date

    _dir = Path(dir)
    _dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Download 1min bars to directory={_dir}")
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

    item = 'bar_1m'
    item_dir = _dir.joinpath(item)
    item_dir.mkdir(parents=True, exist_ok=True)

    existing_dates = [d.split(".")[0] for d in os.listdir(item_dir)]
    # if item == "return" or item == "lag_return":
    #     existing_dates = sorted(existing_dates)[:-6]
    trading_dates = sorted(set(trading_dates) - set(existing_dates))

    if len(trading_dates) == 0:
        click.echo(f"{item} is up to date {datetime.now().date()}")
        return
    click.echo(
        f"Download {item} to directory={item_dir}, {len(trading_dates)} tasks to be done."
    )

    ray.init(num_cpus=n_jobs, ignore_reinit_error=True, include_dashboard=False)
    @ray.remote(max_calls=2)
    def remote_download(begin, end) -> pl.DataFrame:
        try:
            import datareader as dr
            dr.URL.DB73 = "clickhouse://test_wyw_allread:3794b0c0@10.25.1.73:9000"
        except ImportError:
            raise ImportError("Error: module datareader not found")
        
        df: pl.DataFrame = dr.read(
            dr.meta.StockMinute(
                # columns=None, 
                version="3.1", abbr=True),
            begin=begin,
            end=end,
            df_lib='polars'
        )
        res_list = []
        for d, _df in df.partition_by(["date"], as_dict=True).items():
        # for (d, ), _df in df.partition_by(["date"], as_dict=True).items():
            _df.write_parquet(f"{item_dir}/{d:%Y-%m-%d}.parq")
            res_list.append(d)
        return df
        
    task_ids = []
    n_task_finished = 0
    for exp_id, d in enumerate(trading_dates, 1):
        task_id = remote_download.options(
            name="x",
            num_cpus=1,
        ).remote(begin=d, end=d)
        task_ids.append(task_id)

        if len(task_ids) >= n_jobs:
            dones, task_ids = ray.wait(task_ids, num_returns=1)
            ray.get(dones)
            n_task_finished += 1
            logger.info(f"{n_task_finished} tasks finished.")
    ray.get(task_ids)
    click.echo(f"task {item} done.")


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
def download_univ(
    dir: str,
    begin: str,
    end: str
):
    """Download universe."""
    from pathlib import Path
    from datetime import datetime, timedelta

    import polars as pl
    from pit.utils import any2date

    _dir = Path(dir)
    _dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Download 1min bars to directory={_dir}")
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

    item = 'univ'
    item_dir = _dir.joinpath(item)
    item_dir.mkdir(parents=True, exist_ok=True)

    existing_dates = [d.split(".")[0] for d in os.listdir(item_dir)]
    trading_dates = sorted(set(trading_dates) - set(existing_dates))

    if len(trading_dates) == 0:
        click.echo(f"{item} is up to date {datetime.now().date()}")
        return
    click.echo(
        f"Download {item} to directory={item_dir}, {len(trading_dates)} tasks to be done."
    )

    try:
        import datareader as dr
    except ImportError:
        raise ImportError("Error: module datareader not found")
    

    univs = [
        "univ_research",
        "univ_largemid",
        "sz50",
        "hs300",
        "zz500",
        "zz1000",
        "zz2000",
        "euniv_largemid",
        "euniv_research",
        "euniv_eresearch",
        "univ_full",
        "mktcap",
    ]


    df: pl.DataFrame = dr.read(
        dr.meta.StockUniverse(univs), begin=begin, end=end, df_lib='polars'
    )
    df = df.select(["date", "symbol"] + univs).sort(by=["date", "symbol"])
    for d, _df in df.partition_by(["date"], as_dict=True).items():
    # for (d, ), _df in df.partition_by(["date"], as_dict=True).items():
        _df.write_parquet(f"{item_dir}/{d:%Y-%m-%d}.parq")
        
    click.echo(f"task {item} done.")




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
def download_return(
    dir: str,
    begin: str,
    end: str
):
    """Download universe."""
    from pathlib import Path
    from datetime import datetime, timedelta

    import polars as pl
    from pit.utils import any2date

    _dir = Path(dir)
    _dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Download 1min bars to directory={_dir}")
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

    item = 'return'
    item_dir = _dir.joinpath(item)
    item_dir.mkdir(parents=True, exist_ok=True)

    existing_dates = [d.split(".")[0] for d in os.listdir(item_dir)]
    # trading_dates = sorted(set(trading_dates) - set(existing_dates))
    recent_dates = sorted(gu.tcalendar.get(
        begin=gu.tcalendar.adjust('today', -6), end='today'))
    trading_dates = sorted(set(trading_dates) - set(existing_dates) - set(recent_dates))

    if len(trading_dates) == 0:
        click.echo(f"{item} is up to date {datetime.now().date()}")
        return
    click.echo(
        f"Download {item} to directory={item_dir}, {len(trading_dates)} tasks to be done."
    )

    try:
        import datareader as dr
    except ImportError:
        raise ImportError("Error: module datareader not found")
    

    n_list = [1, 2, 3, 5]

    # delay 5 minutes
    slots = [
        ("0935", "1000"),
        (1005, 1030),
        (1035, 1100),
        (1105, 1130),
        (1305, 1330),
        (1335, 1400),
        (1405, 1430),
        (1435, 1500),
    ]
    ret_slots = {}
    # vwap return
    for slot in slots:
        ret_slots[f"_v2v_{slot[0]}"] = (
            f"vwap_{slot[0]}_{slot[1]}",
            f"vwap_{slot[0]}_{slot[1]}",
        )
    # open-to-open return
    for slot in slots:
        ret_slots[f"_o2o_{slot[0]}"] = (f"close_{slot[0]}", f"close_{slot[0]}")

    df: pl.DataFrame = dr.read(
        dr.m.StockReturnDaily(
            ret_slots, n_days=n_list, abbr=False, future=True
        ),
        begin=begin,
        end=end,
        df_lib='polars'
    )
        
    # get intraday returns
    slots = [
        "0935",
        "1005",
        "1035",
        "1105",
        "1305",
        "1335",
        "1405",
        "1435",
        "1000",
        "1030",
        "1100",
        "1130",
        "1330",
        "1400",
        "1430",
        "1500",
    ]
    df_close: pl.DataFrame = dr.read(
        dr.m.StockMinute(["close"]),
        begin=begin,
        end=end,
        at=slots,
        df_lib='polars',
    )
    df_close = df_close.with_columns(pl.col('time').dt.strftime("%H%M").alias('slot'))
        # df_close["slot"] = df_close["time"].dt.strftime("%H%M")

    df_intra = df_close.pivot(
        values="close", columns="slot", index=["date", "symbol"]
    )
    df_intra = df_intra.with_columns(
        pl.col('1000').truediv(pl.col('0935')).sub(1).alias('0930_30m'),
        pl.col('1030').truediv(pl.col('1005')).sub(1).alias('1000_30m'),
        pl.col('1100').truediv(pl.col('1035')).sub(1).alias('1030_30m'),
        pl.col('1130').truediv(pl.col('1105')).sub(1).alias('1100_30m'),
        pl.col('1330').truediv(pl.col('1305')).sub(1).alias('1300_30m'),
        pl.col('1400').truediv(pl.col('1335')).sub(1).alias('1330_30m'),
        pl.col('1430').truediv(pl.col('1405')).sub(1).alias('1400_30m'),
        pl.col('1500').truediv(pl.col('1435')).sub(1).alias('1430_30m'),
        pl.col('1030').truediv(pl.col('0935')).sub(1).alias('0930_1h'),
        pl.col('1100').truediv(pl.col('1005')).sub(1).alias('1000_1h'),
        pl.col('1130').truediv(pl.col('1035')).sub(1).alias('1030_1h'),
        pl.col('1330').truediv(pl.col('1105')).sub(1).alias('1100_1h'),
        pl.col('1400').truediv(pl.col('1305')).sub(1).alias('1300_1h'),
        pl.col('1430').truediv(pl.col('1335')).sub(1).alias('1330_1h'),
        pl.col('1500').truediv(pl.col('1405')).sub(1).alias('1400_1h'),
    )

    intra_return_cols = [
        "0930_30m",
        "1000_30m",
        "1030_30m",
        "1100_30m",
        "1300_30m",
        "1330_30m",
        "1400_30m",
        "1430_30m",
        "0930_1h",
        "1000_1h",
        "1030_1h",
        "1100_1h",
        "1300_1h",
        "1330_1h",
        "1400_1h",
    ]

    df = df.join(
        df_intra.select(intra_return_cols + ["date", "symbol"]),
        on=["date", "symbol"],
        how="left",
    ).sort(by=["date", "symbol"])

    for d, _df in df.partition_by(["date"], as_dict=True).items():
    # for (d, ), _df in df.partition_by(["date"], as_dict=True).items():
        _df.write_parquet(f"{item_dir}/{d:%Y-%m-%d}.parq")
    # return df
    
    click.echo(f"task {item} done.")

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
def download_lag_return(
    dir: str,
    begin: str,
    end: str
):
    """Download lag return."""
    from pathlib import Path
    from datetime import datetime, timedelta

    import polars as pl
    from pit.utils import any2date

    _dir = Path(dir)
    _dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Download 1min bars to directory={_dir}")
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

    item = 'lag_return'
    item_dir = _dir.joinpath(item)
    item_dir.mkdir(parents=True, exist_ok=True)

    existing_dates = [d.split(".")[0] for d in os.listdir(item_dir)]
    recent_dates = sorted(gu.tcalendar.get(
        begin=gu.tcalendar.adjust('today', -6), end='today'))
    trading_dates = sorted(set(trading_dates) - set(existing_dates) - set(recent_dates))
    

    if len(trading_dates) == 0:
        click.echo(f"{item} is up to date {datetime.now().date()}")
        return
    click.echo(
        f"Download {item} to directory={item_dir}, {len(trading_dates)} tasks to be done."
    )

    try:
        import datareader as dr
    except ImportError:
        raise ImportError("Error: module datareader not found")
    

    n_list = [1, 2, 3, 5]

    # delay 5 minutes
    slots = [
        ("0935", "1000"),
        (1005, 1030),
        (1035, 1100),
        (1105, 1130),
        (1305, 1330),
        (1335, 1400),
        (1405, 1430),
        (1435, 1500),
    ]
    ret_slots = {}
    # vwap return
    for slot in slots:
        ret_slots[f"_v2v_{slot[0]}"] = (
            f"vwap_{slot[0]}_{slot[1]}",
            f"vwap_{slot[0]}_{slot[1]}",
        )
    # open-to-open return
    for slot in slots:
        ret_slots[f"_o2o_{slot[0]}"] = (f"close_{slot[0]}", f"close_{slot[0]}")

    df: pl.DataFrame = dr.read(
        dr.m.StockReturnDaily(
            ret_slots, n_days=n_list, abbr=False, future=True
        ),
        begin=begin,
        end=end,
        df_lib='polars'
    )
        
    # get intraday returns
    slots = [
        "0935",
        "1005",
        "1035",
        "1105",
        "1305",
        "1335",
        "1405",
        "1435",
        "1000",
        "1030",
        "1100",
        "1130",
        "1330",
        "1400",
        "1430",
        "1500",
    ]
    df_close: pl.DataFrame = dr.read(
        dr.m.StockMinute(["close"]),
        begin=begin,
        end=end,
        at=slots,
        df_lib='polars',
    )
    df_close = df_close.with_columns(pl.col('time').dt.strftime("%H%M").alias('slot'))
        # df_close["slot"] = df_close["time"].dt.strftime("%H%M")

    df_intra = df_close.pivot(
        values="close", columns="slot", index=["date", "symbol"]
    )
    df_intra = df_intra.with_columns(
        pl.col('1000').truediv(pl.col('0935')).sub(1).alias('0930_30m'),
        pl.col('1030').truediv(pl.col('1005')).sub(1).alias('1000_30m'),
        pl.col('1100').truediv(pl.col('1035')).sub(1).alias('1030_30m'),
        pl.col('1130').truediv(pl.col('1105')).sub(1).alias('1100_30m'),
        pl.col('1330').truediv(pl.col('1305')).sub(1).alias('1300_30m'),
        pl.col('1400').truediv(pl.col('1335')).sub(1).alias('1330_30m'),
        pl.col('1430').truediv(pl.col('1405')).sub(1).alias('1400_30m'),
        pl.col('1500').truediv(pl.col('1435')).sub(1).alias('1430_30m'),
        pl.col('1030').truediv(pl.col('0935')).sub(1).alias('0930_1h'),
        pl.col('1100').truediv(pl.col('1005')).sub(1).alias('1000_1h'),
        pl.col('1130').truediv(pl.col('1035')).sub(1).alias('1030_1h'),
        pl.col('1330').truediv(pl.col('1105')).sub(1).alias('1100_1h'),
        pl.col('1400').truediv(pl.col('1305')).sub(1).alias('1300_1h'),
        pl.col('1430').truediv(pl.col('1335')).sub(1).alias('1330_1h'),
        pl.col('1500').truediv(pl.col('1405')).sub(1).alias('1400_1h'),
    )

    intra_return_cols = [
        "0930_30m",
        "1000_30m",
        "1030_30m",
        "1100_30m",
        "1300_30m",
        "1330_30m",
        "1400_30m",
        "1430_30m",
        "0930_1h",
        "1000_1h",
        "1030_1h",
        "1100_1h",
        "1300_1h",
        "1330_1h",
        "1400_1h",
    ]

    df = df.join(
        df_intra.select(intra_return_cols + ["date", "symbol"]),
        on=["date", "symbol"],
        how="left",
    ).sort(by=["date", "symbol"])

    cols = ["date", "prev", "next"]
    date_lag = gu.tcalendar.getdf(begin=begin, end=end, renew=False)[cols]
    date_lag = pl.from_dataframe(date_lag)
    date_lag = date_lag.with_columns(pl.col('date').cast(pl.Date).alias('date'))
    df_lag = df.join(date_lag, on="date", how="left")
    df_lag = df_lag.drop(["date", "next"])
    df_lag = df_lag.rename({"prev": "date"})
    cols_rename = {
        col: f"lag_{col}" for col in df.columns if col not in ["date", "symbol"]
    }
    df_lag = df_lag.rename(mapping=cols_rename)
        
    for d, _df in df_lag.partition_by(["date"], as_dict=True).items():
    # for (d, ), _df in df.partition_by(["date"], as_dict=True).items():
        _df.write_parquet(f"{item_dir}/{d:%Y-%m-%d}.parq")
    # return df
    
    click.echo(f"task {item} done.")
    
@click.command()
@click.option(
    "--dir", '_dir',
    default="/data2/private/wangxin/raw2",
    type=click.Path(exists=True),
    help="parent directory of data, a subdirectory named `bar_1m_wide` will be created.",
)
@click.option(
    "--n_jobs", default=10, type=int, help="number of parallel jobs are most."
)
@click.option("--verbose", default=False, type=bool, help="whether to print progress.")
def long2widev2(_dir: str, n_jobs: int, verbose: bool):
    import os
    import re
    from pathlib import Path
    import polars as pl
    import ray
    import itertools

    dir = Path(_dir)
    src_dir = dir.joinpath("bar_1m")
    tgt_dir = dir.joinpath("bar_1m_wide_v2")
    tgt_dir.mkdir(parents=True, exist_ok=True)

    src_dates = [re.findall(r"\d{4}-\d{2}-\d{2}", d)[0] for d in os.listdir(src_dir)]
    tgt_dates = [re.findall(r"\d{4}-\d{2}-\d{2}", d)[0] for d in os.listdir(tgt_dir)]
    trading_dates = sorted(set(src_dates) - set(tgt_dates))
    if verbose is True:
        click.echo(f"{trading_dates=}")
    click.echo(f"Long to wide {len(trading_dates)} tasks to be done.")

    cpu_per_task = 2
    ray.init(num_cpus=n_jobs * cpu_per_task, ignore_reinit_error=True, include_dashboard=False)
    @ray.remote(max_calls=1)
    def long2wide_fn(date: str):
        df = pl.scan_parquet(f"{src_dir}/{date}.parq")
        # cols = [c for c in df.columns if c not in ["date", "symbol", "time"]]
        cols = get_bars('v2')
        df = df.with_columns(pl.col("time").dt.strftime("%H%M").alias("slot"))
        df = df.with_columns([pl.col(_c).cast(pl.Float32) for _c in cols]).collect()
        slots = df.get_column("slot").unique().sort().to_list()
        df = df.pivot(index=["symbol", "date"], columns="slot", values=cols)
        name_mapping = {
            f"{col}_slot_{slt}": f"{col}_{slt}"
            for col, slt in itertools.product(cols, slots)
        }
        df = df.rename(name_mapping)
        df.write_parquet(f"{tgt_dir}/{date}.parq")
        return date

    task_ids = []
    n_task_finished = 0
    for exp_id, _d in enumerate(trading_dates, 1):
        if verbose is True:
            click.echo(f"running on {_d}")
        task_id = long2wide_fn.options(
            name="x",
            num_cpus=cpu_per_task,
        ).remote(date=_d)
        task_ids.append(task_id)

        if len(task_ids) >= n_jobs:
            dones, task_ids = ray.wait(task_ids, num_returns=1)
            ray.get(dones)
            n_task_finished += 1
            logger.info(f"{n_task_finished} tasks finished.")
    ray.get(task_ids)
    
    click.echo("task long2wide_v2 done.")
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

pit.add_command(train_single)
pit.add_command(inference)
pit.add_command(show)
pit.add_command(download_1m)
pit.add_command(download_univ)
pit.add_command(download_return)
pit.add_command(download_lag_return)
pit.add_command(long2widev2)

if __name__ == "__main__":
    pit()