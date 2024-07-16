from typing import Optional

import polars as pl

from pit.dr_context import DatareaderContext
from pit.tcalendar import last_day_of_year
from pit.utils import Datetime

__all__ = ["download_stock_minute", "download_universe", "download_tcalendar"]


def download_stock_minute(begin: Datetime, end: Datetime) -> pl.DataFrame:
  """Download all stock minute bars from clickhouse.

  Args:
      begin (Datetime): begin date.
      end (Datetime): end date.

  Raises:
      ImportError: Error: module datareader not found

  Returns:
      pl.DataFrame: _description_
  """
  with DatareaderContext() as dr:
    df: pl.DataFrame = dr.read(
      dr.meta.StockMinute(
        # columns=None,
        version="2",
        abbr=True,
      ),
      begin=begin,
      end=end,
      df_lib="polars",
      categorical_symbol=True,
    )

    # df = df.with_columns(pl.col(pl.NUMERIC_DTYPES).cast(pl.Float32))

    df_factor: pl.DataFrame = dr.read(
      dr.meta.StockDaily(columns=["adj_factor"]),
      begin=begin,
      end=end,
      df_lib="polars",
      categorical_symbol=True,
    )
  df = df.join(df_factor, on=["date", "symbol"], how="left")
  return df


def download_universe(begin: Datetime, end: Datetime) -> pl.DataFrame:
  """Get universe from clickhouse.

  Args:
      begin (Datetime): begin date.
      end (Datetime): end date.

  Raises:
      ImportError: Error: module datareader not found

  Returns:
      pl.DataFrame: universe.
  """

  with DatareaderContext() as dr:
    df: pl.DataFrame = dr.read(
      dr.meta.StockUniverse(),
      begin=begin,
      end=end,
      df_lib="polars",
      categorical_symbol=True,
    )
  df = df.sort(by=["date", "symbol"])
  return df


def download_tcalendar(
  begin: Datetime = "20150101", end: Optional[Datetime] = None
) -> pl.Series:
  """Download trading calendar from external data source.

  Args:
      begin (Datetime): begin date.
      end (Datetime): end date, default to last day of year.

  Raises:
      ImportError: Error: module genutils not found

  Returns:
      pl.DataFrame: trading calendar.
  """
  try:
    import genutils as gu
  except ImportError:
    raise ImportError("Error: module genutils not found")

  _end = last_day_of_year() if end is None else end
  _df = gu.tcalendar.getdf(begin=begin, end=_end)
  calendar_series: pl.Series = pl.from_pandas(_df["date"])
  calendar_series = calendar_series.cast(pl.Date).sort().unique(maintain_order=True)
  return calendar_series


def download_return(begin: Datetime, end: Datetime) -> pl.DataFrame:
  """Download all stock minute bars from clickhouse.

  Args:
      begin (Datetime): begin date.
      end (Datetime): end date.

  Raises:
      ImportError: Error: module datareader not found

  Returns:
      pl.DataFrame: _description_
  """
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

  with DatareaderContext() as dr:
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

  df_intra = df_close.pivot(on="slot", values="close", index=["date", "symbol"])
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
  return df


def download_lag_return(begin: Datetime, end: Datetime) -> pl.DataFrame:
  """Download all stock minute bars from clickhouse.

  Args:
      begin (Datetime): begin date.
      end (Datetime): end date.

  Raises:
      ImportError: Error: module genutils not found

  Returns:
      pl.DataFrame: _description_
  """
  try:
    import genutils as gu
  except ImportError:
    raise ImportError("Error: module genutils not found.")

  df: pl.DataFrame = download_return(begin, end)
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
  return df_lag


def download_ohlcv_minute(begin: Datetime, end: Datetime) -> pl.DataFrame:
  """Download all stock minute bars from clickhouse.

  Args:
      begin (DateType): begin date.
      end (DateType): end date.

  Raises:
      ImportError: Error: module datareader not found

  Returns:
      pl.DataFrame: _description_
  """
  with DatareaderContext() as dr:
    df: pl.DataFrame = dr.read(
      dr.meta.StockMinute(columns=["open", "high", "low", "close", "volume"]),
      begin=begin,
      end=end,
      df_lib="polars",
      categorical_symbol=True,
    )

    # limit 娑ㄥ仠浠? stopping 璺屽仠浠? trade_status 浜ゆ槗鐘舵€?
    df_factor: pl.DataFrame = dr.read(
      dr.meta.StockDaily(columns=["adj_factor", "limit", "stopping"]),
      begin=begin,
      end=end,
      df_lib="polars",
      categorical_symbol=True,
    )
  df = df.join(df_factor, on=["date", "symbol"], how="left")
  df = df.with_columns(
    pl.col("open").mul(pl.col("adj_factor")).alias("adj_open"),
    pl.col("high").mul(pl.col("adj_factor")).alias("adj_high"),
    pl.col("low").mul(pl.col("adj_factor")).alias("adj_low"),
    pl.col("close").mul(pl.col("adj_factor")).alias("adj_close"),
  )
  df = df.with_columns(pl.col(pl.NUMERIC_DTYPES).cast(pl.Float32))
  return df


def download_stock_tick(begin: Datetime, end: Datetime) -> pl.DataFrame:
  """Download all stock tick (snapshot) bars.

  Args:
      begin (DateType): begin date.
      end (DateType): end date.

  Raises:
      ImportError: Error: module datareader not found

  Returns:
      pl.DataFrame: _description_
  """
  with DatareaderContext() as dr:
    df: pl.DataFrame = dr.read(
      dr.meta.StockSnapshot(),
      begin=begin,
      end=end,
      df_lib="polars",
      categorical_symbol=True,
    )
  return df


def download_barra(begin: Datetime, end: Datetime) -> pl.DataFrame:
  """Download daily barra style and industry factors.

  Args:
      begin (DateType): begin date.
      end (DateType): end date.

  Raises:
      ImportError: Error: module datareader not found

  Returns:
      pl.DataFrame: _description_
  """
  with DatareaderContext() as dr:
    df: pl.DataFrame = dr.read(
      dr.meta.StockBarraFactor(abbr=True, wide=True, ignore_country=True),
      begin=begin,
      end=end,
      df_lib="polars",
      categorical_symbol=True,
    )
  return df
