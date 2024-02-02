from typing import TypeVar
import datetime

import dateutil.parser


Datetime = TypeVar('Datetime', str, datetime.datetime)

def normalize_date(dt: Datetime) -> datetime.datetime:
    """Normalize input to be a datetime.datetime object if possible.

    Args:
        dt (Datetime): input date.

    Raises:
        TypeError: input argument type is not one of (str, datetime.datetime).

    Returns:
        datetime.datetime: output object.
    """    
    if isinstance(dt, str):
        o = dateutil.parser.parse(dt)
    elif isinstance(dt, datetime.datetime):
        o = dt
    else:
        raise TypeError(f"dt type {type(dt)} not supported.")
    return o