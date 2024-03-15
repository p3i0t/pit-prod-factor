import datetime
import os
from pathlib import Path

import click
from loguru import logger
from omegaconf import OmegaConf
import polars as pl

try:
    import datareader as dr
except ImportError:
    raise ImportError("Error: module datareader not found.")

try:
    import genutils as gu
except ImportError:
    raise ImportError("Error: module genutils not found.")

from pit import (
    list_prods,
    get_bars,
    get_training_config,
    get_inference_config,
    TrainPipeline,
)
from pit.utils import any2ymd, any2date
from pit.download import (
    download_stock_minute,
    download_universe,
    download_ohlcv_minute,
    download_tcalendar,
)
from pit.config import read_config


tasks_dict = {
    "ohlcv_1m": download_ohlcv_minute,
    "univ": download_universe,
}

# self-defined click ParamType by overriding the convert() method
# to convert the value (as str) to the desired type (date str)
class ExtendedDate(click.ParamType):
    name = "date"

    def convert(self, value, param, ctx):
        try:
            return any2ymd(str(value))
        except ValueError:
            self.fail(f"{value} is not a valid date or 'today'.", param, ctx)


# Instantiate the custom type to use with the click option
DateType = ExtendedDate()

cfg = read_config()

@click.command()
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
@click.option("--n_jobs", default=10, type=int, help="number of ray parallel jobs.")
@click.option("--mem_per_task", default=10, type=int, help="memory per task in GB.")
@click.option("--verbose", "-v", is_flag=True, help="whether to print details.")
def download_1m(begin, end, n_jobs, mem_per_task, verbose):
    """Download 1min bars up to today."""
    import ray
    if gu.tcalendar.trading(begin):
        _begin = any2date(begin)
    else:
        _begin: datetime.date = gu.tcalendar.adjust(begin, 1).date()
        if verbose is True:
            click.echo(f"begin date {begin} is not trading day, adjust to {_begin}")

    if gu.tcalendar.trading(end) and datetime.datetime.now().strftime("%H%M") > "2300":
        _end = any2date(end)
        if verbose is True:
            click.echo(
                f"end date {end} is trading day and data is available at now (till 2330), adjust to {_end}"
            )
    else:
        _end: datetime.date = gu.tcalendar.adjust(end, -1).date()
        if verbose is True:
            click.echo(f"end date {end} is not trading day, adjust to {_end}")

    if _begin <= _end:
        pass
    else:
        click.echo(
            f"begin date {_begin} is later than end date {_end}, no need to download."
        )
        return

    trading_dates = sorted(gu.tcalendar.get(begin=begin, end=_end))
    trading_dates = [d.strftime("%Y-%m-%d") for d in trading_dates]

    if verbose is True:
        click.echo(
            f"Targeted {len(trading_dates)} tasks from {trading_dates[0]} to {trading_dates[-1]}."
        )

    item = "bar_1m"
    item_dir = cfg.raw.bar_1m.dir

    existing_dates = [d.split(".")[0] for d in os.listdir(item_dir)]
    left_dates = sorted(set(trading_dates) - set(existing_dates))

    if len(left_dates) == 0:
        click.echo(f"{item} is up to date {datetime.datetime.now().date()}")
        return
    if verbose is True:
        click.echo(f"Download {item} to directory={item_dir}")
        click.echo(f"{len(trading_dates)} tasks in total")
        click.echo(f"{len(existing_dates)} tasks already exist.")
        click.echo(f"{len(left_dates)} tasks to be done.")

    ray.init(num_cpus=n_jobs, ignore_reinit_error=True, include_dashboard=False)

    @ray.remote(max_calls=1, memory=mem_per_task * 1024 * 1024 * 1024)
    def remote_download(begin, end) -> None:
        df = download_stock_minute(begin, end)
        if df.is_empty():
            if verbose is True:
                click.echo(f"task {begin} is empty.")
            return
        res_list = []
        for d, _df in df.partition_by(["date"], as_dict=True).items():
            # for (d, ), _df in df.partition_by(["date"], as_dict=True).items():
            if verbose is True:
                click.echo(f"task {d:%Y-%m-%d} done.")
            _df.write_parquet(f"{item_dir}/{d:%Y-%m-%d}.parq")
            res_list.append(d)
        # return df

    task_ids = []
    n_task_finished = 0
    for exp_id, d in enumerate(left_dates, 1):
        task_id = remote_download.options(
            name="x",
            num_cpus=1,
        ).remote(begin=d, end=d)
        task_ids.append(task_id)

        if len(task_ids) >= n_jobs:
            dones, task_ids = ray.wait(task_ids, num_returns=1)
            ray.get(dones)
            n_task_finished += 1
            if verbose is True and n_task_finished % 10 == 0:
                click.echo(f"{n_task_finished} tasks finished.")
    ray.get(task_ids)
    if verbose is True:
        click.echo(f"{len(left_dates)} tasks done.")


@click.command()
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
    "--task",
    "-t",
    "task_name",
    default="ohlcv_1m",
    type=click.Choice(list(tasks_dict.keys())),
)
@click.option("--verbose", "-v", is_flag=True, help="whether to print details.")
def download(begin, end, task_name, verbose):
    """Daily Download Tasks."""
    if gu.tcalendar.trading(begin):
        _begin = any2date(begin)
    else:
        _begin: datetime.date = gu.tcalendar.adjust(begin, 1).date()
        if verbose is True:
            click.echo(f"begin date {begin} is not trading day, adjust to {_begin}")

    if gu.tcalendar.trading(end) and datetime.datetime.now().strftime("%H%M") > "2300":
        _end = any2date(end)
        if verbose is True:
            click.echo(
                f"end date {end} is trading day and data is available at now (after 2330), adjust to {_end}"
            )
    else:
        _end: datetime.date = gu.tcalendar.adjust(end, -1).date()
        if verbose is True:
            click.echo(f"end date {end} adjust to {_end}")

    if _begin <= _end:
        pass
    else:
        click.echo(
            f"begin date {_begin} is later than end date {_end}, no need to download."
        )
        return

    trading_dates = sorted(gu.tcalendar.get(begin=begin, end=_end))
    trading_dates = [d.strftime("%Y-%m-%d") for d in trading_dates]

    item = task_name
    item_dir = _dir.joinpath(item)
    item_dir.mkdir(parents=True, exist_ok=True)

    existing_dates = [d.split(".")[0] for d in os.listdir(item_dir)]
    left_dates = sorted(set(trading_dates) - set(existing_dates))

    if len(left_dates) == 0:
        click.echo(f"{item} is up to date {datetime.datetime.now().date()}")
        return
    if verbose is True:
        click.echo(f"Download {item} to directory={item_dir}")
        click.echo(f"data of {len(trading_dates)} dates in total")
        click.echo(f"data of {len(existing_dates)} dates already exist.")
        click.echo(f"data of {len(left_dates)} dates to be done.")

    df = tasks_dict[task_name](begin, end)
    if df.is_empty():
        click.echo("univ dataframe is empty.")
        return
    for d, _df in df.partition_by(["date"], as_dict=True).items():
        # for (d, ), _df in df.partition_by(["date"], as_dict=True).items():
        _df.write_parquet(f"{item_dir}/{d:%Y-%m-%d}.parq")
    click.echo(f"task {task_name} done.")


@click.command()
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
@click.option("--verbose", "-v", is_flag=True, help="whether to print details.")
def download_return(begin, end, verbose):
    """Download return."""
    cfg = read_config()
    _dir = Path(cfg.raw.dir)
    _dir.mkdir(parents=True, exist_ok=True)

    if gu.tcalendar.trading(begin):
        _begin = any2date(begin)
    else:
        _begin: datetime.date = gu.tcalendar.adjust(begin, 1).date()
        if verbose is True:
            click.echo(f"begin date {begin} is not trading day, adjust to {_begin}")

    if gu.tcalendar.trading(end) and datetime.datetime.now().strftime("%H%M") > "2300":
        _end = any2date(end)
        if verbose is True:
            click.echo(
                f"end date {end} is trading day and data is available at now (till 2330), adjust to {_end}"
            )
    else:
        _end: datetime.date = gu.tcalendar.adjust(end, -1).date()
        if verbose is True:
            click.echo(f"end date {end} is not trading day, adjust to {_end}")

    if _begin <= _end:
        pass
    else:
        click.echo(
            f"begin date {_begin} is later than end date {_end}, no need to download."
        )
        return

    trading_dates = sorted(gu.tcalendar.get(begin=begin, end=_end))
    trading_dates = [d.strftime("%Y-%m-%d") for d in trading_dates]

    item = "return"
    item_dir = _dir.joinpath(item)
    item_dir.mkdir(parents=True, exist_ok=True)

    existing_dates = [d.split(".")[0] for d in os.listdir(item_dir)]
    # trading_dates = sorted(set(trading_dates) - set(existing_dates))
    recent_dates = sorted(
        gu.tcalendar.get(begin=gu.tcalendar.adjust("today", -6), end="today")
    )
    left_dates = sorted(set(trading_dates) - set(existing_dates) - set(recent_dates))

    if len(left_dates) == 0:
        click.echo(f"{item} is up to date {datetime.datetime.now().date()}")
        return
    if verbose is True:
        click.echo(f"Download {item} to directory={item_dir}")
        click.echo(f"{len(trading_dates)} dates in total")
        click.echo(f"{len(existing_dates)} dates already exist.")
        click.echo(f"{len(left_dates)} dates to be done.")

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
        dr.m.StockReturnDaily(ret_slots, n_days=n_list, abbr=False, future=True),
        begin=begin,
        end=end,
        df_lib="polars",
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
        df_lib="polars",
    )
    df_close = df_close.with_columns(pl.col("time").dt.strftime("%H%M").alias("slot"))
    # df_close["slot"] = df_close["time"].dt.strftime("%H%M")

    df_intra = df_close.pivot(values="close", columns="slot", index=["date", "symbol"])
    df_intra = df_intra.with_columns(
        pl.col("1000").truediv(pl.col("0935")).sub(1).alias("0930_30m"),
        pl.col("1030").truediv(pl.col("1005")).sub(1).alias("1000_30m"),
        pl.col("1100").truediv(pl.col("1035")).sub(1).alias("1030_30m"),
        pl.col("1130").truediv(pl.col("1105")).sub(1).alias("1100_30m"),
        pl.col("1330").truediv(pl.col("1305")).sub(1).alias("1300_30m"),
        pl.col("1400").truediv(pl.col("1335")).sub(1).alias("1330_30m"),
        pl.col("1430").truediv(pl.col("1405")).sub(1).alias("1400_30m"),
        pl.col("1500").truediv(pl.col("1435")).sub(1).alias("1430_30m"),
        pl.col("1030").truediv(pl.col("0935")).sub(1).alias("0930_1h"),
        pl.col("1100").truediv(pl.col("1005")).sub(1).alias("1000_1h"),
        pl.col("1130").truediv(pl.col("1035")).sub(1).alias("1030_1h"),
        pl.col("1330").truediv(pl.col("1105")).sub(1).alias("1100_1h"),
        pl.col("1400").truediv(pl.col("1305")).sub(1).alias("1300_1h"),
        pl.col("1430").truediv(pl.col("1335")).sub(1).alias("1330_1h"),
        pl.col("1500").truediv(pl.col("1405")).sub(1).alias("1400_1h"),
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
    click.echo(f"task {item} done.")


@click.command()
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
@click.option("--verbose", "-v", is_flag=True, help="whether to print details.")
def download_lag_return(dir, begin, end, verbose):
    """Download lag return."""
    # from datetime import datetime, timedelta
    cfg = read_config()
    _dir = Path(cfg.raw.dir)
    _dir.mkdir(parents=True, exist_ok=True)

    if gu.tcalendar.trading(begin):
        _begin = any2date(begin)
    else:
        _begin: datetime.date = gu.tcalendar.adjust(begin, 1).date()
        if verbose is True:
            click.echo(f"begin date {begin} is not trading day, adjust to {_begin}")

    if gu.tcalendar.trading(end) and datetime.datetime.now().strftime("%H%M") > "2300":
        _end = any2date(end)
        if verbose is True:
            click.echo(
                f"end date {end} is trading day and data is available at now (till 2330), adjust to {_end}"
            )
    else:
        _end: datetime.date = gu.tcalendar.adjust(end, -1).date()
        if verbose is True:
            click.echo(f"end date {end} is not trading day, adjust to {_end}")

    if _begin <= _end:
        pass
    else:
        click.echo(
            f"begin date {_begin} is later than end date {_end}, no need to download."
        )
        return

    trading_dates = sorted(gu.tcalendar.get(begin=begin, end=_end))
    trading_dates = [d.strftime("%Y-%m-%d") for d in trading_dates]

    item = "return"
    item_dir = _dir.joinpath(item)
    item_dir.mkdir(parents=True, exist_ok=True)

    existing_dates = [d.split(".")[0] for d in os.listdir(item_dir)]
    # trading_dates = sorted(set(trading_dates) - set(existing_dates))
    recent_dates = sorted(
        gu.tcalendar.get(begin=gu.tcalendar.adjust("today", -6), end="today")
    )
    left_dates = sorted(set(trading_dates) - set(existing_dates) - set(recent_dates))

    if len(left_dates) == 0:
        click.echo(f"{item} is up to date {datetime.datetime.now().date()}")
        return
    if verbose is True:
        click.echo(f"Download {item} to directory={item_dir}")
        click.echo(f"{len(trading_dates)} dates in total")
        click.echo(f"{len(existing_dates)} dates already exist.")
        click.echo(f"{len(left_dates)} dates to be done.")

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
        dr.m.StockReturnDaily(ret_slots, n_days=n_list, abbr=False, future=True),
        begin=begin,
        end=end,
        df_lib="polars",
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
        df_lib="polars",
    )
    df_close = df_close.with_columns(pl.col("time").dt.strftime("%H%M").alias("slot"))
    # df_close["slot"] = df_close["time"].dt.strftime("%H%M")

    df_intra = df_close.pivot(values="close", columns="slot", index=["date", "symbol"])
    df_intra = df_intra.with_columns(
        pl.col("1000").truediv(pl.col("0935")).sub(1).alias("0930_30m"),
        pl.col("1030").truediv(pl.col("1005")).sub(1).alias("1000_30m"),
        pl.col("1100").truediv(pl.col("1035")).sub(1).alias("1030_30m"),
        pl.col("1130").truediv(pl.col("1105")).sub(1).alias("1100_30m"),
        pl.col("1330").truediv(pl.col("1305")).sub(1).alias("1300_30m"),
        pl.col("1400").truediv(pl.col("1335")).sub(1).alias("1330_30m"),
        pl.col("1430").truediv(pl.col("1405")).sub(1).alias("1400_30m"),
        pl.col("1500").truediv(pl.col("1435")).sub(1).alias("1430_30m"),
        pl.col("1030").truediv(pl.col("0935")).sub(1).alias("0930_1h"),
        pl.col("1100").truediv(pl.col("1005")).sub(1).alias("1000_1h"),
        pl.col("1130").truediv(pl.col("1035")).sub(1).alias("1030_1h"),
        pl.col("1330").truediv(pl.col("1105")).sub(1).alias("1100_1h"),
        pl.col("1400").truediv(pl.col("1305")).sub(1).alias("1300_1h"),
        pl.col("1430").truediv(pl.col("1335")).sub(1).alias("1330_1h"),
        pl.col("1500").truediv(pl.col("1405")).sub(1).alias("1400_1h"),
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
    date_lag = date_lag.with_columns(pl.col(c).cast(pl.Date).alias(c) for c in cols)
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
    click.echo(f"task {item} done.")


@click.command()
@click.option(
    "--n_jobs", default=10, type=int, help="number of parallel jobs are most."
)
@click.option(
    "--cpus_per_task", "n_cpu", default=2, type=int, help="number of cpus per task."
)
@click.option("--verbose", "-v", is_flag=True, help="whether to print progress.")
def long2widev2(n_jobs, n_cpu, verbose):
    """Long to wide for v2 bars."""
    import re
    import ray
    import itertools

    cfg = read_config()
    dir = Path(cfg.raw.dir)
    src_dir = dir.joinpath("bar_1m")
    tgt_dir = dir.joinpath("bar_1m_wide_v2")
    if verbose is True:
        click.echo(f"long2wide from {src_dir} to {tgt_dir}")
    tgt_dir.mkdir(parents=True, exist_ok=True)

    src_dates = [re.findall(r"\d{4}-\d{2}-\d{2}", d)[0] for d in os.listdir(src_dir)]
    tgt_dates = [re.findall(r"\d{4}-\d{2}-\d{2}", d)[0] for d in os.listdir(tgt_dir)]
    trading_dates = sorted(set(src_dates) - set(tgt_dates))
    if verbose is True:
        click.echo(f"{trading_dates=}")
    click.echo(f"Long to wide {len(trading_dates)} tasks to be done.")

    ray.init(num_cpus=n_jobs * n_cpu, ignore_reinit_error=True, include_dashboard=False)

    @ray.remote(max_calls=1)
    def long2wide_fn(date: str):
        if verbose is True:
            click.echo(f"running on {date}")
        df = pl.scan_parquet(f"{src_dir}/{date}.parq")
        # cols = [c for c in df.columns if c not in ["date", "symbol", "time"]]
        cols = get_bars("v2")
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
            num_cpus=n_cpu,
        ).remote(date=_d)
        task_ids.append(task_id)

        if len(task_ids) >= n_jobs:
            dones, task_ids = ray.wait(task_ids, num_returns=1)
            ray.get(dones)
            n_task_finished += 1
            logger.info(f"{n_task_finished} tasks finished.")
    ray.get(task_ids)
    click.echo("task long2wide_v2 done.")


@click.command()
@click.option("--n_jobs", default=10, type=int, help="number of parallel jobs.")
@click.option(
    "--cpu_per_task", "n_cpu", default=4, type=int, help="number of cpus per task."
)
@click.option("--verbose", "-v", is_flag=True, help="whether to print progress.")
def downsample10(n_jobs, n_cpu, verbose):
    """Downsample 1m to 10m."""
    import glob
    cfg = read_config()
    src_dir = Path(cfg.raw.dir).joinpath("bar_1m")
    tgt_dir = Path(cfg.derived_dir).joinpath("bar_10m")
    from pit.downsample import downsample_1m_to_10m

    if verbose is True:
        click.echo(f"downsample from {src_dir} to {tgt_dir}")
    Path(tgt_dir).mkdir(parents=True, exist_ok=True)
    import ray
    import re

    @ray.remote(max_calls=3)
    def _downsample(file):
        bars = get_bars("v3")
        downsample_1m_to_10m(
            pl.scan_parquet(f"{src_dir}/{file}"), bars=bars
        ).write_parquet(f"{tgt_dir}/{file}")
        return file

    src_files = set(glob.glob(f"{src_dir}/*.parq"))
    tgt_files = set(glob.glob(f"{tgt_dir}/*.parq"))

    src_dates = set([re.findall(r"\d{4}-\d{2}-\d{2}", d)[0] for d in src_files])
    tgt_dates = set([re.findall(r"\d{4}-\d{2}-\d{2}", d)[0] for d in tgt_files])
    left_dates = sorted(src_dates - tgt_dates)
    click.echo(f"{len(left_dates)} downsample tasks to be done.")
    left_files = [f"{d}.parq" for d in left_dates]

    task_ids = []
    n_task_finished = 0
    for exp_id, file in enumerate(left_files, 1):
        if verbose is True:
            click.echo(f"running on {file}")
        task_id = _downsample.options(
            name="x",
            num_cpus=n_cpu,
        ).remote(file=file)
        task_ids.append(task_id)

        if len(task_ids) >= n_jobs:
            dones, task_ids = ray.wait(task_ids, num_returns=1)
            ray.get(dones)
            n_task_finished += 1
            if verbose is True and n_task_finished % 10 == 0:
                click.echo(f"{n_task_finished} tasks finished.")
    ray.get(task_ids)
    click.echo("task downsample done.")


@click.command()
@click.option(
    "--n_jobs", default=20, type=int, help="number of parallel jobs are most."
)
@click.option(
    "--cpu_per_task", "n_cpu", default=2, type=int, help="number of cpus per task."
)
def merge10_v2(n_jobs, n_cpu):
    dir_10m = "/data2/private/wangxin/raw2/bar_10m"
    dir_univ = "/data2/private/wangxin/raw2/univ"
    dir_ret = "/data2/private/wangxin/raw2/return"
    dir_lag_ret = "/data2/private/wangxin/raw2/lag_return"

    tgt_dir = Path("/data2/private/wangxin/dataset/10m_v2")
    tgt_dir.mkdir(parents=True, exist_ok=True)
    from collections import defaultdict
    import re
    import ray
    import itertools
    from dlkit.utils import get_time_slots

    bars = get_bars("v2")
    slots = get_time_slots("0930", "1500", freq_in_min=10)
    v2_factors = [
        f"{_b}_{_agg}" for _b, _agg in itertools.product(bars, ["mean", "std"])
    ]
    v2_cols = [f"{_f}_{_s}" for _f, _s in itertools.product(v2_factors, slots)]

    src_files = (
        set(os.listdir(dir_10m)) & set(os.listdir(dir_univ)) & set(os.listdir(dir_ret))
    )
    src_dates = [re.findall(r"\d{4}-\d{2}-\d{2}", d)[0] for d in sorted(src_files)[:-6]]

    monthly_groups = defaultdict(list)
    for _date in src_dates:
        monthly_groups[re.findall(r"\d{4}-\d{2}", _date)[0]].append(_date)

    @ray.remote(max_calls=3)
    def _merge(month, dates):
        with pl.StringCache():
            s, t = min(dates), max(dates)
            exists = [file for file in os.listdir(tgt_dir) if file.startswith(month)]

            if len(exists) == 0:
                # df_cur = pl.DataFrame()  # empty df
                pass
            elif len(exists) == 1:
                cur_s, cur_t = re.findall(r"\d{4}-\d{2}-\d{2}", exists[0])
                if all(cur_s <= _d <= cur_t for _d in dates):
                    return  # already merged
                else:
                    os.remove(f"{tgt_dir}/{exists[0]}")
                    logger.info(f"remove {tgt_dir}/{exists[0]}")
            else:
                for _file in exists:
                    os.remove(f"{tgt_dir}/{_file}")
                    logger.info(f"remove {tgt_dir}/{_file}")
                # raise ValueError(f"multiple files found for {month}")

            df_list = []
            for _d in dates:
                df_univ = pl.scan_parquet(f"{dir_univ}/{_d}.parq").collect()
                df_10m = (
                    pl.scan_parquet(f"{dir_10m}/{_d}.parq")
                    .select(["date", "symbol"] + v2_cols)
                    .collect()
                )
                df_10m = df_10m.with_columns(pl.col("symbol").cast(pl.Categorical))
                df_ret = pl.scan_parquet(f"{dir_ret}/{_d}.parq").collect()
                df_lag_ret = pl.scan_parquet(f"{dir_lag_ret}/{_d}.parq").collect()
                df = df_univ.join(df_10m, on=["date", "symbol"], how="left")
                df = df.join(df_ret, on=["date", "symbol"], how="left")
                df = df.join(df_lag_ret, on=["date", "symbol"], how="left")
                df_list.append(df)
            df_final = pl.concat(df_list)
            cols = df_final.select(pl.col(pl.Float64)).columns
            df_final = df_final.with_columns(pl.col(c).cast(pl.Float32) for c in cols)
            df_final.write_parquet(f"{tgt_dir}/{s}_{t}.parq")
        return month

    task_ids = []
    n_task_finished = 0
    for _month, _dates in monthly_groups.items():
        task_id = _merge.options(
            name="x",
            num_cpus=n_cpu,
        ).remote(month=_month, dates=_dates)
        task_ids.append(task_id)

        if len(task_ids) >= n_jobs:
            dones, task_ids = ray.wait(task_ids, num_returns=1)
            ray.get(dones)
            n_task_finished += 1
            logger.info(f"{n_task_finished} tasks finished.")
    ray.get(task_ids)


@click.command()
@click.option(
    "--verbose", "-v", is_flag=True, help="whether to print more information."
)
def update_tcalendar(verbose):
    """Update trading calendar."""
    cfg = read_config()
    tclendar_series = download_tcalendar()
    if verbose:
        click.echo(f"Updating calendar to {cfg.calendar_path}")
        click.echo(
            f"{len(tclendar_series)} dates from {tclendar_series[0]} to {tclendar_series[-1]}."
        )
    tclendar_series.to_frame().write_csv(cfg.calendar_path)


@click.command()
@click.option(
    "--prod", "-p", default=click.Choice(list_prods()), help="product to infer"
)
@click.option(
    "--milestone",
    "-m",
    type=DateType,
    default="today",
    help="milestone date of the model to train.",
)
def train_single(prod, milestone):
    """Train single model of given prod and milestone."""
    args = get_training_config(prod=prod, milestone=milestone)
    pipe = TrainPipeline(args)
    pipe.run()


@click.command()
@click.option(
    "--prod", "-p", default=click.Choice(list_prods()), help="product to infer"
)
@click.option(
    "--mode",
    "-m",
    default="train",
    type=click.Choice(["train", "inference"]),
    help="train or inference args to show.",
)
def show(prod, mode):
    """List training or inference args."""
    if mode == "train":
        args = get_training_config(prod=prod)
    elif mode == "inference":
        args = get_inference_config(prod=prod)
    else:
        raise ValueError(f"mode {mode} not in ['train', 'inference']")
    print(args)


@click.command()
@click.option(
    "--prod", "-p", default=click.Choice(list_prods()), help="product to infer"
)
@click.option(
    "--date", "-d", default="today", help="the date of data used for inference."
)
@click.option(
    "--verbose", "-v", is_flag=True, help="whether to print more information."
)
def infer_online(prod, date, verbose):
    """Online inference on single date.

    For prod used at 0930 of next trading day, the date in the result is the next trading day after infer_date.
    """
    import sys
    from pit.inference import infer, InferenceMode
    from datetime import timedelta

    infer_date = any2date(date)
    args = get_inference_config(prod=prod)

    if len(gu.tcalendar.get(infer_date, infer_date)) == 0:
        print(f"generate date {infer_date} is not a trading date !!!")
        sys.exit(0)

    o = infer(
        args=args, infer_date=infer_date, mode=InferenceMode.online, verbose=verbose
    )
    assert isinstance(o, pl.DataFrame)
    if prod in ["0930", "0930_1h"]:
        next_date = gu.tcalendar.adjust(infer_date, 1).date()
        # replace date with next_date
        o = o.with_columns(pl.lit(next_date).cast(pl.Date).alias("date"))

    slot = args.y_slots
    assert isinstance(slot, str)
    o = o.with_columns(
        pl.col("date")
        .cast(pl.Datetime(time_unit="ns"))
        .add(timedelta(hours=int(slot[:2]), minutes=int(slot[2:])))
        .alias("time")
    )

    pit_dir = os.path.join(os.getenv("PIT_HOME", os.path.expanduser("~")), ".pit")
    infer_dir = Path(OmegaConf.load(open(f"{pit_dir}/config.yml")).INFER_DIR)
    tgt_dir = infer_dir.joinpath(prod)
    tgt_dir.mkdir(parents=True, exist_ok=True)

    alpha = o.select(["date", "time", "symbol", args.tgt_column]).rename(
        mapping={args.tgt_column: "pit"}
    )
    use_date = next_date if prod in ["0930", "0930_1h"] else infer_date
    alpha.write_parquet(tgt_dir.joinpath(f"{use_date}.parq"))


@click.command()
@click.option(
    "--prod", "-p", default=click.Choice(list_prods()), help="product to infer"
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
    "--verbose", is_flag=True, help="print more information when debug is True."
)
def infer_hist(prod, begin, end, verbose):
    """Inference on historical data.
    For prod used at 0930 of next trading day, the date in the result is the next trading day after infer_date.
    """
    from pit.inference import infer, InferenceMode
    from datetime import timedelta

    args = get_inference_config(prod=prod)
    begin = any2date(begin)
    end = any2date(end)
    o = infer(
        args=args, infer_date=(begin, end), mode=InferenceMode.offline, verbose=verbose
    )
    assert isinstance(o, pl.DataFrame)
    if prod in ["0930", "0930_1h"]:
        date_lag = gu.tcalendar.getdf(begin, end)
        date_lag = pl.from_pandas(date_lag)
        date_lag = date_lag.with_columns(
            pl.col(c).cast(pl.Date) for c in ["date", "next", "prev"]
        )
        o = o.join(date_lag, on="date", how="left")
        o = o.drop(["date", "prev"])
        o = o.rename({"next": "date"})

    slot = args.y_slots
    assert isinstance(slot, str)
    o = o.with_columns(
        pl.col("date")
        .cast(pl.Datetime(time_unit="ns"))
        .add(timedelta(hours=int(slot[:2]), minutes=int(slot[2:])))
        .alias("time")
    )

    pit_dir = os.path.join(os.getenv("PIT_HOME", os.path.expanduser("~")), ".pit")
    infer_dir = Path(OmegaConf.load(open(f"{pit_dir}/config.yml")).INFER_DIR)
    tgt_dir = infer_dir.joinpath(prod)
    tgt_dir.mkdir(parents=True, exist_ok=True)

    use_begin = (
        gu.tcalendar.adjust(begin, 1).date() if prod in ["0930", "0930_1h"] else begin
    )
    use_end = gu.tcalendar.adjust(end, 1).date() if prod in ["0930", "0930_1h"] else end

    o.write_parquet(tgt_dir.joinpath(f"hist_pred_{use_begin}_{use_end}.parq"))
    alpha = o.select(["date", "time", "symbol", args.tgt_column]).rename(
        mapping={args.tgt_column: "alpha"}
    )
    alpha.write_parquet(tgt_dir.joinpath(f"hist_alpha_{use_begin}_{use_end}.parq"))


@click.command()
def init():
    from pit.config import init_config
    home_dir = init_config()
    click.echo(f'Initialize config as {home_dir}')

    
@click.command()
def show_config():
    cfg = read_config()
    click.echo(cfg)


@click.group()
def pit():
    """pit: Alpha Signals Generator of Pit."""
    # click.echo("Alpha Signals Generator of Pit.")


pit.add_command(train_single)
pit.add_command(show)
pit.add_command(download)
pit.add_command(download_1m)
pit.add_command(download_return)
pit.add_command(download_lag_return)
pit.add_command(long2widev2)
pit.add_command(downsample10)
pit.add_command(merge10_v2)
pit.add_command(infer_hist)
pit.add_command(infer_online)
pit.add_command(update_tcalendar)
pit.add_command(init)
pit.add_command(show_config)

if __name__ == "__main__":
    pit()
