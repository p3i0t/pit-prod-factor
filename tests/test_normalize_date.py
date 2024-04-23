import datetime
import pytest

from pit.utils import normalize_date


@pytest.mark.parametrize(
    "dt,expected",
    [
        (datetime.datetime(2023,1,9), datetime.datetime(2023,1,9)),
        ('2023-01-09', datetime.datetime(2023,1,9)),
        ('2024-01-09 09:30:18', datetime.datetime(2024,1,9,9,30,18))
    ]
)
def test_get_time_slots(dt, expected):
    assert expected == normalize_date(dt)