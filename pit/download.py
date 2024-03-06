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
        df_lib='polars'
    )
    return df