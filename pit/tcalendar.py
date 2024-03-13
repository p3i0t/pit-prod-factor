import bisect
import datetime
from functools import lru_cache
import os
import polars as pl
from pit.utils import any2ymd, Datetime


def _load_tcalendar() -> list[str]:
    """Load trading calendar from file.

    Returns:
        dict: trading calendar.
    """
    pit_dir = os.path.join(
        os.getenv("PIT_HOME", os.path.expanduser("~")), ".pit")
    
    calendar_path = f"{pit_dir}/tcalendar.csv"
    return pl.read_csv(calendar_path).get_column('date').to_list()


def is_string_in_list(sorted_list: list[str], target: str) -> bool:
    """
    Checks if a target string is in the sorted list.
    Returns True if the target is found, otherwise False.
    """
    index = bisect.bisect_left(sorted_list, target)
    if index != len(sorted_list) and sorted_list[index] == target:
        return True
    return False


def is_trading_day(date: Datetime) -> bool:
    """Check if a date is a trading day.

    Args:
        date (Datetime): date.

    Returns:
        bool: True if it is a trading day.
    """
    return is_string_in_list(_load_tcalendar(), any2ymd(date))


@lru_cache(maxsize=10)
def get_tcalendar_df(n_shift: int) -> pl.DataFrame:
    """Get trading calendar dataframe.

    Args:
        n_shift (int): shift.

    Returns:
        pl.DataFrame: trading calendar dataframe.
    """
    tcalendar = _load_tcalendar()
    df = pl.DataFrame({
        'date': tcalendar,
    })
    df = df.with_columns(
        pl.col('date').cast(pl.Date),
    )
    df = df.with_columns(pl.col('date').shift(n_shift).alias('date_shift'))
    return df


def adjust(date: Datetime, n_shift: int) -> datetime.date:
    """Adjust date with shift.

    Args:
        date (Datetime): date.
        n_shift (int): shift.

    Returns:
        Datetime: adjusted date.
    """
    _date = any2ymd(date)
    tcalendar_list = _load_tcalendar()
    ii = bisect.bisect_left(tcalendar_list, _date)
    jj = ii + n_shift
    if jj < 0 or jj >= len(tcalendar_list):
        raise IndexError(f"Index out of calendar: {jj}")
    return datetime.datetime.strptime(tcalendar_list[jj], "%Y-%m-%d").date()