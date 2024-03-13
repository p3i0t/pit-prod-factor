import datetime
from typing import Optional
import polars as pl

from pit.utils import Datetime


def get_stock_minute(begin: Datetime, end: Datetime) -> pl.DataFrame:
    """Get all stock minute bars from clickhouse.

    Args:
        begin (Datetime): begin date.
        end (Datetime): end date.

    Raises:
        ImportError: Error: module datareader not found

    Returns:
        pl.DataFrame: _description_
    """
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
        df_lib='polars',
        categorical_symbol=True,
    )
    
    # df = df.with_columns(pl.col(pl.NUMERIC_DTYPES).cast(pl.Float32))
    
    df_factor: pl.DataFrame = dr.read(
        dr.meta.StockDaily(columns=['adj_factor']),
        begin=begin,
        end=end,
        df_lib='polars',
        categorical_symbol=True,
    )
    df = df.join(df_factor, on=['date', 'symbol'], how='left')
    return df

    
def get_ohlcv_minute(begin: Datetime, end: Datetime) -> pl.DataFrame:
    """Get all stock minute bars from clickhouse.

    Args:
        begin (Datetime): begin date.
        end (Datetime): end date.

    Raises:
        ImportError: Error: module datareader not found

    Returns:
        pl.DataFrame: _description_
    """
    try:
        import datareader as dr
        dr.URL.DB73 = "clickhouse://test_wyw_allread:3794b0c0@10.25.1.73:9000"
    except ImportError:
        raise ImportError("Error: module datareader not found")
    
    df: pl.DataFrame = dr.read(
        dr.meta.StockMinute(
            columns=['open', 'high', 'low', 'close', 'volume']),
        begin=begin,
        end=end,
        df_lib='polars',
        categorical_symbol=True,
    )
    
    df = df.with_columns(pl.col(pl.NUMERIC_DTYPES).cast(pl.Float32))
    
    df_factor: pl.DataFrame = dr.read(
        dr.meta.StockDaily(columns=['adj_factor']),
        begin=begin,
        end=end,
        df_lib='polars',
        categorical_symbol=True,
    )
    df = df.join(df_factor, on=['date', 'symbol'], how='left')
    return df


def get_universe(begin: Datetime, end: Datetime) -> pl.DataFrame:
    """Get universe from clickhouse.

    Args:
        begin (Datetime): begin date.
        end (Datetime): end date.

    Raises:
        ImportError: Error: module datareader not found

    Returns:
        pl.DataFrame: universe.
    """
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
        dr.meta.StockUniverse(univs), begin=begin, end=end, df_lib='polars', categorical_symbol=True
    )
    df = df.select(["date", "symbol"] + univs).sort(by=["date", "symbol"])
    return df



def last_day_of_year(year: Optional[int] = None) -> datetime.date:
    if year is None:
        year = datetime.datetime.now().year
    return datetime.date(year, 12, 31)


def download_tcalendar(begin: Datetime = '20150101', end: Optional[Datetime] = None) -> pl.Series:
    """Download trading calendar from external data source.

    Args:
        begin (Datetime): begin date.
        end (Datetime): end date.

    Raises:
        ImportError: Error: module datareader not found

    Returns:
        pl.DataFrame: trading calendar.
    """
    try:
        import genutils as gu
    except ImportError:
        raise ImportError("Error: module datareader not found")

    _end = last_day_of_year() if end is None else end
    _df = gu.tcalendar.getdf(begin=begin, end=_end)
    calendar_series: pl.Series = pl.from_pandas(_df['date'])
    calendar_series = calendar_series.cast(pl.Date).sort().unique(maintain_order=True)
    return calendar_series