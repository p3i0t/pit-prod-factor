import datetime
import glob
import itertools
import os
from collections import defaultdict
from pathlib import Path
from time import perf_counter

import click
import polars as pl
import polars.selectors as cs
import ray
from loguru import logger

from pit import (
  InferencePipeline,
  TrainPipeline,
  __version__,
  get_bars,
  get_inference_config,
  get_training_config,
  list_prods,
)
from pit.config import read_config
from pit.download import (
  download_lag_return,
  download_return,
  download_stock_minute,
  download_tcalendar,
  download_universe,
)
from pit.tcalendar import (
  adjust_date,
  get_tcalendar_df,
  is_trading_day,
  load_tcalendar_list,
)
from pit.utils import any2date, any2ymd

# data of these tasks is stored as one file per date.
tasks_dict = {
  "bar_1m": download_stock_minute,
  "univ": download_universe,
  "return": download_return,
  "lag_return": download_lag_return,
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


class TaskNotSupportedError(Exception): ...


def _download_single_date_bar_1m(
  # *,
  task_name: str,
  tasks_dict: dict,
  item_dir: str,
  verbose: bool,
  _date: str,
) -> str | None:
  "keep it as a simple python function."
  df = tasks_dict[task_name](_date, _date)
  if df.is_empty():
    if verbose is True:
      click.echo(f"task {_date} is empty.")
    return
  df.write_parquet(f"{item_dir}/{_date}.parq")
  return _date


def _run_download_for_one_task(
  begin,
  end,
  task_name: str,
  verbose: bool = True,
  n_jobs: int = 10,
  cpus_per_task: int = 2,
):
  if task_name not in tasks_dict:
    click.echo(f"task {task_name} is not supported.")
    raise TaskNotSupportedError(f"task {task_name} is not one of {tasks_dict.keys()}.")
  if is_trading_day(begin):
    _begin = any2date(begin)
  else:
    _begin: datetime.date = adjust_date(begin, 1)
    if verbose is True:
      click.echo(f"begin date {begin} is not trading day, adjust to {_begin}")

  if is_trading_day(end) and datetime.datetime.now().strftime("%H%M") > "2300":
    _end = any2date(end)
    if verbose is True:
      click.echo(
        f"end date {end} is trading day and data is available at now (till 2330), adjust to {_end}"
      )
  else:
    _end: datetime.date = adjust_date(end, -1)
    if verbose is True:
      click.echo(f"end date {end} is not trading day, adjust to {_end}")

  if _begin > _end:
    click.echo(
      f"begin date {_begin} is later than end date {_end}, no need to download."
    )
    return

  trading_dates = sorted(load_tcalendar_list(begin=_begin, end=_end))
  if verbose is True:
    click.echo(
      f"task {task_name}: {len(trading_dates)} jobs from {trading_dates[0]} to {trading_dates[-1]}."
    )

  cfg = read_config()
  item_dir = Path(cfg.raw.dir).joinpath(task_name)

  existing_dates = sorted([d.split(".")[0] for d in os.listdir(item_dir)])
  if task_name in ["return", "lag_return"]:
    n_recent = 6
    existing_dates = existing_dates[:-n_recent]
  left_dates = sorted(set(trading_dates) - set(existing_dates))

  if len(left_dates) == 0:
    click.echo(f"{task_name} is up to date {datetime.datetime.now().date()}")
    return
  if verbose is True:
    click.echo(f"Download {task_name} to directory={item_dir}")
    click.echo(f"{len(trading_dates)} tasks in total")
    click.echo(f"{len(existing_dates)} tasks already exist.")
    click.echo(f"{len(left_dates)} tasks to be done.")

  if task_name == "bar_1m":
    task_ids = []
    n_task_finished = 0
    for exp_id, d in enumerate(left_dates, 1):
      task_id = (
        ray.remote(_download_single_date_bar_1m)
        .options(name="x", num_cpus=cpus_per_task)
        .remote(task_name, tasks_dict, str(item_dir), verbose, d)
      )
      task_ids.append(task_id)
      if len(task_ids) >= n_jobs:
        dones, task_ids = ray.wait(task_ids, num_returns=1)
        ray.get(dones)
        n_task_finished += 1
        if verbose is True and n_task_finished % 10 == 0:
          click.echo(f"{n_task_finished} tasks finished.")
    ray.get(task_ids)

    click.echo(f"{len(left_dates)} tasks done.")
  else:
    # fetch all in one run since data is small.
    df = tasks_dict[task_name](left_dates[0], left_dates[-1])
    if df.is_empty():
      click.echo("dataframe is empty.")
      return
    for (d,), _df in df.partition_by(["date"], as_dict=True).items():
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
@click.option(
  "--task",
  "-t",
  "task_name",
  default="ohlcv_1m",
  type=click.Choice(["return", "lag_return", "univ", "bar_1m", "all"]),
)
@click.option("--verbose", "-v", is_flag=True, help="whether to print details.")
@click.option(
  "--n_jobs",
  default=10,
  type=int,
  help="num of jobs in parallel (for bar_1m only), defaults to 10.",
)
@click.option(
  "--cpu_per_task",
  "n_cpu",
  default=2,
  type=int,
  help="num of cpus per task (for bar_1m only), defaults to 2.",
)
def download(begin, end, task_name, verbose, n_jobs, n_cpu):
  """Download data via datareader."""
  if task_name == "all":
    tasks = list(tasks_dict.keys())
  else:
    if task_name not in tasks_dict:
      click.echo(f"task {task_name} is not supported.")
      raise TaskNotSupportedError(
        f"task {task_name} is not one of {tasks_dict.keys()}."
      )
    tasks = [task_name]

  s = perf_counter()
  for task in tasks:
    click.echo(f"\U0001f680 {task=}.")
    _run_download_for_one_task(
      begin, end, task, verbose=verbose, n_jobs=n_jobs, cpus_per_task=n_cpu
    )

  t = perf_counter() - s
  click.echo(f"all tasks done in {t:.2f}s.")


@click.command()
@click.option("--n_jobs", default=10, type=int, help="number of parallel jobs.")
@click.option(
  "--cpu_per_task", "n_cpu", default=4, type=int, help="number of cpus per task."
)
@click.option("--verbose", "-v", is_flag=True, help="whether to print progress.")
def downsample10(n_jobs, n_cpu, verbose):
  """Downsample 1m to 10m."""
  cfg = read_config()
  src_dir = Path(cfg.raw.dir).joinpath("bar_1m")
  tgt_dir = Path(cfg.derived.dir).joinpath("bar_10m")
  from pit.downsample import downsample_1m_to_10m

  if verbose is True:
    click.echo(f"downsample from {src_dir} to {tgt_dir}")
  Path(tgt_dir).mkdir(parents=True, exist_ok=True)
  import re

  @ray.remote(max_calls=3)
  def _downsample(file):
    bars = get_bars("v3")
    downsample_1m_to_10m(pl.scan_parquet(f"{src_dir}/{file}"), bars=bars).write_parquet(
      f"{tgt_dir}/{file}"
    )
    return file

  src_files = set(glob.glob(f"{src_dir}/*.parq"))
  tgt_files = set(glob.glob(f"{tgt_dir}/*.parq"))

  src_dates = set([re.findall(r"\d{4}-\d{2}-\d{2}", d)[0] for d in src_files])
  tgt_dates = set([re.findall(r"\d{4}-\d{2}-\d{2}", d)[0] for d in tgt_files])
  left_dates = sorted(src_dates - tgt_dates)
  click.echo(f"{len(left_dates)} downsample tasks to be done.")
  left_files = [f"{d}.parq" for d in left_dates]

  s = perf_counter()
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
  t = perf_counter() - s
  click.echo(f"all tasks done in {t:.2f}s.")


def _merge_single_group(group_tag: str, group_dates: str | list[str]):
  if isinstance(group_dates, str):
    group_dates = [group_dates]

  cfg = read_config()
  dir_1m = Path(cfg.raw.dir).joinpath("bar_1m")
  dir_univ = Path(cfg.raw.dir).joinpath("univ")
  dir_ret = Path(cfg.raw.dir).joinpath("return")
  dir_lag_ret = Path(cfg.raw.dir).joinpath("lag_return")

  tgt_dir = Path(cfg.dataset.dir).joinpath("10m_v2")
  if not tgt_dir.exists():
    tgt_dir.mkdir(parents=True, exist_ok=True)
  # clear year and month group
  if len(group_tag) == 4 or len(group_tag) == 7:
    date_stems = [
      Path(_p).stem for _p in glob.glob(os.path.join(tgt_dir, f"{group_tag}*.parq"))
    ]
    for stem in date_stems:
      if stem != group_tag:
        os.remove(os.path.join(tgt_dir, f"{stem}.parq"))  # small files are replaced.
    if os.path.exists(os.path.join(tgt_dir, f"{group_tag}.parq")):
      return

  from dlkit.utils import get_time_slots

  from pit.downsample import downsample_1m_to_10m

  bars = get_bars("v2")
  slots = get_time_slots("0930", "1500", freq_in_min=10)
  v2_factors = [f"{_b}_{_agg}" for _b, _agg in itertools.product(bars, ["mean", "std"])]
  v2_cols = [f"{_f}_{_s}" for _f, _s in itertools.product(v2_factors, slots)]

  with pl.StringCache():
    df_list = []
    for _date in group_dates:
      df_univ = pl.scan_parquet(f"{dir_univ}/{_date}.parq").collect()
      bars = get_bars("v2")
      df_10m = downsample_1m_to_10m(
        pl.read_parquet(f"{dir_1m}/{_date}.parq"), bars=bars
      )
      df_10m = df_10m.select(["date", "symbol"] + v2_cols)
      df_10m = df_10m.with_columns(pl.col("symbol").cast(pl.Categorical))
      df_ret = pl.scan_parquet(f"{dir_ret}/{_date}.parq").collect()
      df_lag_ret = pl.scan_parquet(f"{dir_lag_ret}/{_date}.parq").collect()
      df = df_univ.join(df_10m, on=["date", "symbol"], how="left")
      df = df.join(df_ret, on=["date", "symbol"], how="left")
      df = df.join(df_lag_ret, on=["date", "symbol"], how="left")
      df = df.with_columns(cs.numeric().cast(pl.Float32))
      # df.write_parquet(f"{tgt_dir}/{_date}.parq")
      df_list.append(df)
    df = pl.concat(df_list)
    df.write_parquet(f"{tgt_dir}/{group_tag}.parq")


@click.command()
@click.option(
  "--n_jobs", default=10, type=int, help="num of parallel jobs, defaults to 10."
)
@click.option(
  "--cpu_per_task",
  "n_cpu",
  default=2,
  type=int,
  help="number of cpus per task, defaults to 2.",
)
def generate_dataset(n_jobs, n_cpu):
  """Generate dataset from downloaded raw data."""
  cfg = read_config()
  dir_1m = Path(cfg.raw.dir).joinpath("bar_1m")
  dir_univ = Path(cfg.raw.dir).joinpath("univ")
  dir_ret = Path(cfg.raw.dir).joinpath("return")
  dir_lag_ret = Path(cfg.raw.dir).joinpath("lag_return")

  dates_1m = [Path(_p).stem for _p in glob.glob(os.path.join(dir_1m, "*.parq"))]
  dates_univ = [Path(_p).stem for _p in glob.glob(os.path.join(dir_univ, "*.parq"))]
  dates_ret = [Path(_p).stem for _p in glob.glob(os.path.join(dir_ret, "*.parq"))]
  dates_lag_ret = [
    Path(_p).stem for _p in glob.glob(os.path.join(dir_lag_ret, "*.parq"))
  ]

  src_dates = set(dates_1m) & set(dates_univ) & set(dates_ret) & set(dates_lag_ret)
  # always drop 6 recent dates to avoid incomplete data.
  src_dates = sorted(src_dates)[:-6]

  # merge dates as much as possible
  this_year = datetime.datetime.now().year
  this_month = datetime.datetime.now().month

  year_groups = defaultdict(list)
  for _date in src_dates:
    if int(_date[:4]) != this_year:
      year_groups[_date[:4]].append(_date)
  month_groups = defaultdict(list)
  for _date in src_dates:
    if int(_date[:4]) == this_year and int(_date[5:7]) < this_month:
      month_groups[_date[:7]].append(_date)
  date_groups = defaultdict(list)
  for _date in src_dates:
    if int(_date[:4]) == this_year and int(_date[5:7]) == this_month:
      date_groups[_date].append(_date)
  
  all_groups = year_groups | month_groups | date_groups
  click.echo(f"{len(src_dates)} dates in total.")
  click.echo(f"{len(all_groups)} tasks (groups).")

  s = perf_counter()
  task_ids = []
  # n_task_finished = 0
  for group_tag, group_dates in all_groups.items():
    task_id = (
      ray.remote(_merge_single_group)
      .options(
        name="x",
        num_cpus=n_cpu,
      )
      .remote(group_tag, group_dates)
    )
    task_ids.append(task_id)

    if len(task_ids) >= n_jobs:
      dones, task_ids = ray.wait(task_ids, num_returns=1)
      ray.get(dones)
      # n_task_finished += 1
      # logger.info(f"{n_task_finished} tasks finished.")
  ray.get(task_ids)
  t = perf_counter() - s
  click.echo(f"{len(all_groups)} tasks done in {t:.2f}s.")


@click.command()
@click.option(
  "--prod",
  "-p",
  default="1030",
  type=click.Choice(list_prods()),
  help="product name, defaults to `1030`.",
)
@click.option(
  "--milestone",
  "-m",
  type=DateType,
  default="today",
  help="milestone date of the model to train, defaults to `today`.",
)
@click.option(
  "--universe",
  "-u",
  default="euniv_largemid",
  help="universe name, defaults to `euniv_largemid`.",
)
def train_single(prod, milestone, universe):
  """Train single model of given prod and milestone."""
  s = perf_counter()
  args = get_training_config(prod=prod, milestone=milestone, universe=universe)
  pipe = TrainPipeline(args)
  pipe.run()
  t = perf_counter() - s
  click.echo(f"train {prod} on milestone {milestone} done in {t:.2f}s.")


@click.command()
@click.option(
  "--prod",
  "-p",
  default="1030",
  type=click.Choice(list_prods()),
  help="product name, defaults to `1030`.",
)
@click.option(
  "--date",
  "-d",
  default="today",
  help="the date of data used for inference, defaults to `today`.",
)
@click.option(
  "--n_latest",
  default=1,
  type=int,
  help="number of latest models to use, defaults to 1.",
)
@click.option(
  "--universe",
  "-u",
  default="euniv_largemid",
  help="universe name, defaults to `euniv_largemid`.",
)
@click.option("--out-dir", "-o", default=None, help="output directory.")
@click.option("--verbose", "-v", is_flag=True, help="print more information.")
def infer_online(prod, date, n_latest, universe, out_dir, verbose):
  """Online inference on single date.

  For prod used at 0930 of next trading day, the date in the result is the next trading day after infer_date.
  """
  import sys
  from datetime import timedelta

  from pit.inference import InferenceMode, infer

  infer_date = any2date(date)
  args = get_inference_config(prod=prod, n_latest=n_latest, universe=universe)
  if not is_trading_day(infer_date):
    click.echo(f"generate date {infer_date} is not a trading date !!!")
    sys.exit(0)

  o = infer(
    args=args, infer_date=infer_date, mode=InferenceMode.online, verbose=verbose
  )

  assert isinstance(o, pl.DataFrame)
  if prod in ["0930", "0930_1h"]:
    next_date = adjust_date(infer_date, 1)
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

  cfg = read_config()
  infer_dir = Path(cfg.infer_dir)
  tgt_dir = infer_dir.joinpath(prod)
  tgt_dir.mkdir(parents=True, exist_ok=True)
  o = o.select(["date", "time", "symbol", args.tgt_column]).rename(
    mapping={args.tgt_column: "alpha"}
  )

  n_valid_values = (
    o.select([pl.col("alpha").is_not_nan() & pl.col("alpha").is_not_null()])
    .sum()
    .item()
  )

  if isinstance(o, pl.DataFrame):
    click.echo(
      f"infer on {args.universe}, {len(o)} symbols, {n_valid_values} real valid values."
    )

  use_date = next_date if prod in ["0930", "0930_1h"] else infer_date

  out_dir = Path(out_dir) if out_dir else tgt_dir
  out_dir.mkdir(parents=True, exist_ok=True)
  o.write_parquet(out_dir.joinpath(f"{use_date}.parq"))


@click.command()
@click.option(
  "--prod",
  "-p",
  default="1030",
  type=click.Choice(list_prods()),
  help="product name.",
)
@click.option(
  "--begin",
  "-b",
  default="2017-01-01",
  type=DateType,
  help="begin date, e.g. '20210101', '2021-01-01', 'today'.",
)
@click.option(
  "--end",
  "-e",
  default="today",
  type=DateType,
  help="end date, e.g. '20231001', '2023-10-01', or `today`.",
)
@click.option("--n_latest", default=1, type=int, help="number of latest models to use.")
@click.option("--universe", "-u", default="euniv_largemid", help="universe name.")
@click.option("--out-dir", "-o", default=None, help="output directory.")
def infer_hist(prod, begin, end, n_latest, universe, out_dir):
  """Inference on historical data.
  For prod used at 0930 of next trading day, the date in the result is the next trading day after infer_date.
  """
  # from pit.inference import infer, InferenceMode
  from datetime import timedelta

  args = get_inference_config(prod=prod, n_latest=n_latest, universe=universe)

  # _begin = any2ymd(begin)
  # _end = any2ymd(end)

  # all_dates = [d.split(".")[0] for d in os.listdir(args.dataset_dir)]
  # infer_dates = [d for d in all_dates if _begin <= d <= _end]

  s = perf_counter()
  ip = InferencePipeline(args=args)

  from pit.datasource import OfflineDataSource
  with pl.StringCache():
    ds = OfflineDataSource(
        data_path=str(args.dataset_dir) + "/*",
        columns=["date", "symbol"] + args.x_slot_columns,
        universe=args.universe,
        date_range=(any2date(begin), any2date(end)),
        date_col="date",
        fill_nan=True,
      )
    o = ip(ds.collect())
  # with pl.StringCache():
  #   df_list = []
  #   for infer_date in infer_dates:
  #     df: pl.LazyFrame = pl.scan_parquet(f"{args.dataset_dir}/{infer_date}.parq")
  #     if args.universe:
  #       df = df.filter(pl.col(args.universe))
  #     # must be behind universe filter, because `universe` column is not in `columns``.
  #     df = df.select(["date", "symbol"] + args.x_slot_columns)
  #     df = df.with_columns(pl.col(pl.NUMERIC_DTYPES).fill_nan(pl.lit(None)))
  #     df_list.append(ip(df.collect()))

  #   o = pl.concat(df_list)

  assert isinstance(o, pl.DataFrame)
  if prod in ["0930", "0930_1h"]:
    date_lag = get_tcalendar_df(n_next=1).filter(
      (pl.col("date") >= any2date(begin)) & (pl.col("date") <= any2date(end))
    )
    date_lag = date_lag.with_columns(pl.col(c).cast(pl.Date) for c in ["date", "next"])
    o = o.join(date_lag, on="date", how="left")
    o = o.drop(["date"])
    o = o.rename({"next": "date"})

  slot = args.y_slots
  assert isinstance(slot, str)
  o = o.with_columns(
    pl.col("date")
    .cast(pl.Datetime(time_unit="ns"))
    .add(timedelta(hours=int(slot[:2]), minutes=int(slot[2:])))
    .alias("time")
  )
  cfg = read_config()
  infer_dir = Path(cfg.infer_dir)
  tgt_dir = infer_dir.joinpath(prod)
  tgt_dir.mkdir(parents=True, exist_ok=True)

  use_begin = adjust_date(begin, 1) if prod in ["0930", "0930_1h"] else begin
  use_end = adjust_date(end, 1) if prod in ["0930", "0930_1h"] else end
  alpha = o.select(["date", "time", "symbol", args.tgt_column]).rename(
    mapping={args.tgt_column: "alpha"}
  )
  out_dir = Path(out_dir) if out_dir else tgt_dir
  out_dir.mkdir(parents=True, exist_ok=True)
  alpha.write_parquet(out_dir.joinpath(f"hist_{use_begin}_{use_end}.parq"))
  t = perf_counter() - s
  click.echo(f"Inference on {args.universe} done in {t:.2f}s.")


@click.group(invoke_without_command=True)
@click.pass_context
@click.version_option(version=__version__, message="%(version)s")
def pit(ctx):
  """Generator of Alpha Signal Pit."""
  if ctx.invoked_subcommand is None:
    click.echo(f"Pit Version: {__version__}")
    click.echo("No command was invoked. Use --help for more information.")


pit.add_command(train_single)
pit.add_command(download)
pit.add_command(generate_dataset)
pit.add_command(infer_hist)
pit.add_command(infer_online)

if __name__ == "__main__":
  pit()
