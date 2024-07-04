import pytest
import datetime
from pit.utils import any2datetime, any2ymd

@pytest.mark.parametrize(
    "ts_input, expected",
    [
        ("20220201", "2022-02-01"),
        ("2022-02-01", "2022-02-01"),
        ("today", any2ymd("today")),
        (20220201, "2022-02-01"),
        (datetime.date(2022, 2, 1), "2022-02-01"),
        (datetime.datetime(2022, 2, 1, 0, 0), "2022-02-01"),
        (datetime.datetime(2022, 2, 1, 12, 30), "2022-02-01"),
    ],
)
def test_any2ymd(ts_input, expected):
    assert any2ymd(ts_input) == expected
    

@pytest.mark.parametrize(
    "ts_input, expected",
    [
        ("20220201", datetime.datetime(2022, 2, 1, 0, 0)),
        ("2022-02-01", datetime.datetime(2022, 2, 1, 0, 0)),
        ("today", datetime.datetime.combine(datetime.datetime.now().date(), datetime.time())),
        (20220201, datetime.datetime(2022, 2, 1, 0, 0)),
        (datetime.date(2022, 2, 1), datetime.datetime(2022, 2, 1, 0, 0)),
        (datetime.datetime(2022, 2, 1, 0, 0), datetime.datetime(2022, 2, 1, 0, 0)),
        (datetime.datetime(2022, 2, 1, 12, 30), datetime.datetime(2022, 2, 1, 12, 30)),
    ],
)
def test_any2datetime(ts_input, expected):
    assert any2datetime(ts_input) == expected
