from typing import TypeVar, Annotated
import datetime

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






