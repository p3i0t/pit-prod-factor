import bisect
import datetime
from functools import lru_cache
import os
import polars as pl
from pit.utils import any2ymd, Datetime, compute_gap_days_and_end_slot

__all__ = ["is_trading_day", "get_tcalendar_df", "adjust_date"]

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


def get_tcalendar_df(n_next: int) -> pl.DataFrame:
    """Get trading calendar dataframe.

    Args:
        n_next (int): number of next days.

    Returns:
        pl.DataFrame: trading calendar dataframe.
    """
    tcalendar = _load_tcalendar()
    df = pl.DataFrame({'date': tcalendar})
    df = df.with_columns(
        pl.col('date').cast(pl.Date),
    )
    df = df.with_columns(pl.col('date').shift(-n_next).alias('next'))
    return df


def adjust_date(date: Datetime, n_shift: int) -> datetime.date:
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
    if tcalendar_list[ii] != _date and n_shift > 0:
        n_shift -= 1
    jj = ii + n_shift
    if jj < 0 or jj >= len(tcalendar_list):
        raise IndexError(f"Index out of calendar: {jj}")
    return datetime.datetime.strptime(tcalendar_list[jj], "%Y-%m-%d").date()


def adjust_tcalendar_slot(slot: str, duration: str) -> pl.DataFrame:
    """Adjust duration with shift.

    Args:
        duration (str): duration.
        n_shift (int): shift.

    Returns:
        str: adjusted duration.
    """
    n_next, end_slot = compute_gap_days_and_end_slot(slot, duration)
    df = get_tcalendar_df(n_next)
    start_duration = pl.duration(hours=int(slot[:2]), minutes=int(slot[2:]))
    end_duration = pl.duration(hours=int(end_slot[:2]), minutes=int(end_slot[2:]))
    df = df.with_columns(
        pl.col('date').cast(pl.Datetime(time_unit='ns')).add(start_duration),
        pl.col('next').cast(pl.Datetime(time_unit='ns')).add(end_duration),
    )
    return df