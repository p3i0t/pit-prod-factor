from typing import TypeVar, Annotated, Tuple, List, Literal
import datetime
from functools import lru_cache
import re

import dateutil.parser

__all__ = ['any2datetime', 'any2date', 'any2ymd', 'Datetime']

Datetime = TypeVar('Datetime', 
                   Annotated[str, "string like '20220201', '2022-02-01' or 'today'."], 
                   datetime.datetime, 
                   datetime.date,
                   Annotated[int, "int like 20220201."])

def any2datetime(dt: Datetime) -> datetime.datetime:
    if isinstance(dt, datetime.datetime):
        return dt
    elif isinstance(dt, datetime.date):
        return datetime.datetime.combine(dt, datetime.time())
    if isinstance(dt, int):
        return dateutil.parser.parse(str(dt))
    if isinstance(dt, str):
        if dt == 'today':
            o = datetime.datetime.combine(
                    datetime.datetime.now().date(), datetime.time()
                )
        else:
            o = dateutil.parser.parse(dt)
        return o
    else:
        raise TypeError(f"dt type {type(dt)} not supported.")

def any2date(ts_input) -> datetime.date:
    return any2datetime(ts_input).date()

def any2ymd(ts_input) -> str:
    return any2date(ts_input).strftime("%Y-%m-%d")


@lru_cache(maxsize=128)
def parse_dhm(dhm: str) -> Tuple[int, int]:
    """parse a duration string like '1d2h3m' to a tuple of days and minutes

    Args:
        dhm (str): input duration string.

    Raises:
        ValueError: _description_
        ValueError: _description_

    Returns:
        Tuple[int, int]: number of days and minutes.
    """
    # Define the conversion factors to minutes
    conversion_factors = {'d': 1440, 'h': 60, 'm': 1}
    
    # Check for any characters that are not digits or allowed units
    if not re.fullmatch(r"(\d+d)?(\d+h)?(\d+m)?", dhm):
        raise ValueError("Invalid format. Make sure to use the format: 1d2h3m, and each unit only exists once.")
    
    # Find all occurrences of number followed by a duration unit
    matches = re.findall(r'(\d+)([dhm])', dhm)
    
    # Check if each unit appears only once
    units = [match[1] for match in matches]
    if len(units) != len(set(units)):
        raise ValueError("Each unit ('d', 'h', 'm') is only allowed to exist once.")

    total_minutes = 0
    for match in matches:
        number, unit = match
        total_minutes += int(number) * conversion_factors[unit]
    
    if total_minutes == 0:
        raise ValueError("Duration must be greater than 0.")
    
    _d, _m = divmod(total_minutes, 1440)
    return _d, _m


@lru_cache(maxsize=128)
def get_time_slots(
    start: str = "0930",
    end: str = "1030",
    freq_in_min: Literal[1, 10] = 10,
    bar_on_the_right: bool = True,
) -> List[str]:
    """Generate the list of intraday time slots.

    Args:
        start (str, optional): start slot. Defaults to "0930".
        end (str, optional): end slot. Defaults to "1030".
        freq_in_min (Literal[1, 10], optional): number of minutes as stride. Defaults to 10.
        bar_on_the_right (bool, optional): start slot exclusive if True,
            otherwise end slot exclusive. Defaults to True.

    Returns:
        list[str]: list of time slots.

    Examples:
        note that trading time is 0930-1130, 1300-1500.
        >>> get_time_slots(start="0930", end="1030", freq_in_min=10, bar_on_the_right=True)
        ['0940', '0950', '1000', '1010', '1020', '1030']
        >>> get_time_slots(start="0930", end="1030", freq_in_min=10, bar_on_the_right=False)
        ['0930', '0940', '0950', '1000', '1010', '1020']
        >>> get_time_slots(start="1030", end="1300", freq_in_min=10, bar_on_the_right=True)
        ['1040', '1050', '1100', '1110', '1120', '1130']
    """
    _start = datetime.datetime.strptime(start, "%H%M")
    _end = datetime.datetime.strptime(end, "%H%M")

    # A-shares
    morning_start = datetime.datetime.strptime("0930", "%H%M")
    morning_end = datetime.datetime.strptime("1130", "%H%M")
    afternoon_start = datetime.datetime.strptime("1300", "%H%M")
    afternoon_end = datetime.datetime.strptime("1500", "%H%M")

    freq = datetime.timedelta(minutes=freq_in_min)
    slots = []
    while _start <= _end:
        slots.append(_start)
        _start += freq

    if bar_on_the_right is True:
        slots = slots[1:]
        slots = [
            _s
            for _s in slots
            if morning_start < _s <= morning_end
            or afternoon_start < _s <= afternoon_end
        ]
    else:
        slots = slots[:-1]
        slots = [
            _s
            for _s in slots
            if morning_start <= _s < morning_end
            or afternoon_start <= _s < afternoon_end
        ]

    # right end, so 0930, 1300 excluded.
    slots = [f"{_s:%H%M}" for _s in slots]
    return slots


def get_gap_days_and_end(start: str, duration: str) -> Tuple[int, str]:
    """get the end time given start time and duration.

    Args:
        start (str): start time.
        duration (str): duration string like '1d2h3m'.

    Returns:
        str: end time.
    """
    _d, _m = parse_dhm(duration)
    
    slots = get_time_slots(start='0930', end="1500", freq_in_min=1, bar_on_the_right=True)
    
    s_index = slots.index(start)
    n_days, t_idnex = divmod(s_index + _m, len(slots))
    end = slots[t_idnex]
    n_days += _d
    return n_days, end
