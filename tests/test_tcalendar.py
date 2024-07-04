import pytest

from pit.tcalendar import _parse_dhm, compute_gap_days_and_end_slot

@pytest.mark.parametrize(
    "dhm, expected",
    [
        ("1d2h3m", (1, 123)),
        ("1d2h", (1, 120)),
        ("1d3m", (1, 3)),
        ("2h3m", (0, 123)),
        ("1d", (1, 0)),
        ("2h", (0, 120)),
        ("3m", (0, 3)),
    ],
)
def test_parse_dhm(dhm, expected):
    assert _parse_dhm(dhm) == expected

def test_parse_dhm_errors():
    with pytest.raises(ValueError):
        _parse_dhm("1d2h3m3m")
    with pytest.raises(ValueError):
        _parse_dhm("1d2h2h3m")
    with pytest.raises(ValueError):
        _parse_dhm("1d2h3m3h")
    with pytest.raises(ValueError):
        _parse_dhm("1d2h3m3d")
    with pytest.raises(ValueError):
        _parse_dhm("0m")
    with pytest.raises(ValueError):
        _parse_dhm("0d")
    with pytest.raises(ValueError):
        _parse_dhm("0h")
    with pytest.raises(ValueError):
        _parse_dhm("0d0h0m")
    with pytest.raises(ValueError):
        _parse_dhm("0d0h")
    with pytest.raises(ValueError):
        _parse_dhm("0d0m")
    with pytest.raises(ValueError):
        _parse_dhm("0h0m")
        
        
def test_compute_gap_days_and_end_slot():
    assert compute_gap_days_and_end_slot('1030', "1d") == (1, "1030")
    assert compute_gap_days_and_end_slot('1030', "10d") == (10, "1030")
    assert compute_gap_days_and_end_slot('1030', "1d1h") == (1, "1130")
    assert compute_gap_days_and_end_slot('1030', "1d1h1m") == (1, "1301")
    assert compute_gap_days_and_end_slot('1030', "1d1m") == (1, "1031")
    assert compute_gap_days_and_end_slot('1030', "1h") == (0, "1130")
    assert compute_gap_days_and_end_slot('1030', "2h1m") == (0, "1401")
    assert compute_gap_days_and_end_slot('1030', "3h1m") == (1, "0931")
    