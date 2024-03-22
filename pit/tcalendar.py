from typing import Literal, Tuple, Optional, TypeAlias
import bisect
import datetime
# from functools import lru_cache
import os
import polars as pl
from pit.utils import any2ymd, Datetime
from pit.config import read_config

__all__ = ['get_trading_slots', "is_trading_day", "get_tcalendar_df", "adjust_date", "adjust_tcalendar_slot_df"]

# A-share minute trading slots, bar on the right
trading_slots_right = ['0931',
 '0932',
 '0933',
 '0934',
 '0935',
 '0936',
 '0937',
 '0938',
 '0939',
 '0940',
 '0941',
 '0942',
 '0943',
 '0944',
 '0945',
 '0946',
 '0947',
 '0948',
 '0949',
 '0950',
 '0951',
 '0952',
 '0953',
 '0954',
 '0955',
 '0956',
 '0957',
 '0958',
 '0959',
 '1000',
 '1001',
 '1002',
 '1003',
 '1004',
 '1005',
 '1006',
 '1007',
 '1008',
 '1009',
 '1010',
 '1011',
 '1012',
 '1013',
 '1014',
 '1015',
 '1016',
 '1017',
 '1018',
 '1019',
 '1020',
 '1021',
 '1022',
 '1023',
 '1024',
 '1025',
 '1026',
 '1027',
 '1028',
 '1029',
 '1030',
 '1031',
 '1032',
 '1033',
 '1034',
 '1035',
 '1036',
 '1037',
 '1038',
 '1039',
 '1040',
 '1041',
 '1042',
 '1043',
 '1044',
 '1045',
 '1046',
 '1047',
 '1048',
 '1049',
 '1050',
 '1051',
 '1052',
 '1053',
 '1054',
 '1055',
 '1056',
 '1057',
 '1058',
 '1059',
 '1100',
 '1101',
 '1102',
 '1103',
 '1104',
 '1105',
 '1106',
 '1107',
 '1108',
 '1109',
 '1110',
 '1111',
 '1112',
 '1113',
 '1114',
 '1115',
 '1116',
 '1117',
 '1118',
 '1119',
 '1120',
 '1121',
 '1122',
 '1123',
 '1124',
 '1125',
 '1126',
 '1127',
 '1128',
 '1129',
 '1130',
 '1301',
 '1302',
 '1303',
 '1304',
 '1305',
 '1306',
 '1307',
 '1308',
 '1309',
 '1310',
 '1311',
 '1312',
 '1313',
 '1314',
 '1315',
 '1316',
 '1317',
 '1318',
 '1319',
 '1320',
 '1321',
 '1322',
 '1323',
 '1324',
 '1325',
 '1326',
 '1327',
 '1328',
 '1329',
 '1330',
 '1331',
 '1332',
 '1333',
 '1334',
 '1335',
 '1336',
 '1337',
 '1338',
 '1339',
 '1340',
 '1341',
 '1342',
 '1343',
 '1344',
 '1345',
 '1346',
 '1347',
 '1348',
 '1349',
 '1350',
 '1351',
 '1352',
 '1353',
 '1354',
 '1355',
 '1356',
 '1357',
 '1358',
 '1359',
 '1400',
 '1401',
 '1402',
 '1403',
 '1404',
 '1405',
 '1406',
 '1407',
 '1408',
 '1409',
 '1410',
 '1411',
 '1412',
 '1413',
 '1414',
 '1415',
 '1416',
 '1417',
 '1418',
 '1419',
 '1420',
 '1421',
 '1422',
 '1423',
 '1424',
 '1425',
 '1426',
 '1427',
 '1428',
 '1429',
 '1430',
 '1431',
 '1432',
 '1433',
 '1434',
 '1435',
 '1436',
 '1437',
 '1438',
 '1439',
 '1440',
 '1441',
 '1442',
 '1443',
 '1444',
 '1445',
 '1446',
 '1447',
 '1448',
 '1449',
 '1450',
 '1451',
 '1452',
 '1453',
 '1454',
 '1455',
 '1456',
 '1457',
 '1458',
 '1459',
 '1500']

# A-share minute trading slots, bar on the left
trading_slots_left = ['0930',
 '0931',
 '0932',
 '0933',
 '0934',
 '0935',
 '0936',
 '0937',
 '0938',
 '0939',
 '0940',
 '0941',
 '0942',
 '0943',
 '0944',
 '0945',
 '0946',
 '0947',
 '0948',
 '0949',
 '0950',
 '0951',
 '0952',
 '0953',
 '0954',
 '0955',
 '0956',
 '0957',
 '0958',
 '0959',
 '1000',
 '1001',
 '1002',
 '1003',
 '1004',
 '1005',
 '1006',
 '1007',
 '1008',
 '1009',
 '1010',
 '1011',
 '1012',
 '1013',
 '1014',
 '1015',
 '1016',
 '1017',
 '1018',
 '1019',
 '1020',
 '1021',
 '1022',
 '1023',
 '1024',
 '1025',
 '1026',
 '1027',
 '1028',
 '1029',
 '1030',
 '1031',
 '1032',
 '1033',
 '1034',
 '1035',
 '1036',
 '1037',
 '1038',
 '1039',
 '1040',
 '1041',
 '1042',
 '1043',
 '1044',
 '1045',
 '1046',
 '1047',
 '1048',
 '1049',
 '1050',
 '1051',
 '1052',
 '1053',
 '1054',
 '1055',
 '1056',
 '1057',
 '1058',
 '1059',
 '1100',
 '1101',
 '1102',
 '1103',
 '1104',
 '1105',
 '1106',
 '1107',
 '1108',
 '1109',
 '1110',
 '1111',
 '1112',
 '1113',
 '1114',
 '1115',
 '1116',
 '1117',
 '1118',
 '1119',
 '1120',
 '1121',
 '1122',
 '1123',
 '1124',
 '1125',
 '1126',
 '1127',
 '1128',
 '1129',
 '1300',
 '1301',
 '1302',
 '1303',
 '1304',
 '1305',
 '1306',
 '1307',
 '1308',
 '1309',
 '1310',
 '1311',
 '1312',
 '1313',
 '1314',
 '1315',
 '1316',
 '1317',
 '1318',
 '1319',
 '1320',
 '1321',
 '1322',
 '1323',
 '1324',
 '1325',
 '1326',
 '1327',
 '1328',
 '1329',
 '1330',
 '1331',
 '1332',
 '1333',
 '1334',
 '1335',
 '1336',
 '1337',
 '1338',
 '1339',
 '1340',
 '1341',
 '1342',
 '1343',
 '1344',
 '1345',
 '1346',
 '1347',
 '1348',
 '1349',
 '1350',
 '1351',
 '1352',
 '1353',
 '1354',
 '1355',
 '1356',
 '1357',
 '1358',
 '1359',
 '1400',
 '1401',
 '1402',
 '1403',
 '1404',
 '1405',
 '1406',
 '1407',
 '1408',
 '1409',
 '1410',
 '1411',
 '1412',
 '1413',
 '1414',
 '1415',
 '1416',
 '1417',
 '1418',
 '1419',
 '1420',
 '1421',
 '1422',
 '1423',
 '1424',
 '1425',
 '1426',
 '1427',
 '1428',
 '1429',
 '1430',
 '1431',
 '1432',
 '1433',
 '1434',
 '1435',
 '1436',
 '1437',
 '1438',
 '1439',
 '1440',
 '1441',
 '1442',
 '1443',
 '1444',
 '1445',
 '1446',
 '1447',
 '1448',
 '1449',
 '1450',
 '1451',
 '1452',
 '1453',
 '1454',
 '1455',
 '1456',
 '1457',
 '1458',
 '1459']

ClosedInterval: TypeAlias = Literal['left', 'right', 'both', 'none']

def get_trading_slots(
    begin: Optional[str] = None, 
    end: Optional[str] = None, 
    freq_in_minute: Literal[1, 10] = 1, 
    bar_on_the_right=True,
    closed: ClosedInterval = 'both'
    ) -> list[str]:
    """Get the intraday minute trading slots of A-share.

    Args:
        freq_in_minute (Literal[1, 10], optional): frequency in minutes. Defaults to 1.
        bar_on_the_right (bool, optional): the side of the bar for every minute. Defaults to True.
        closed (Literal['left', 'right', 'both', 'none'], optional): which sides of the interval are closed. Defaults to 'both'.

    Returns:
        list[str]: list of trading slots.
    """
    _begin = begin or "0930"
    _end = end or "1500"
    _slots = trading_slots_right if bar_on_the_right else trading_slots_left
    if freq_in_minute == 1:
        ...
    elif freq_in_minute == 10:
        _slots = _slots[9::10]
    else:
        raise ValueError("freq_in_minute should be 1 or 10.")

    if closed == 'left':
        return [slot for slot in _slots if slot >= _begin and slot < _end]
    elif closed == 'right':
        return [slot for slot in _slots if slot > _begin and slot <= _end]
    elif closed == 'both':
        return [slot for slot in _slots if slot >= _begin and slot <= _end]
    elif closed == 'none':
        return [slot for slot in _slots if slot > _begin and slot < _end]
    


def _parse_dhm(dhm: str) -> Tuple[int, int]:
    """parse a duration string like '1d2h3m' to a tuple of days and minutes

    Args:
        dhm (str): input duration string.

    Raises:
        ValueError: _description_
        ValueError: _description_

    Returns:
        Tuple[int, int]: number of days and minutes.
    """
    import re
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

def last_day_of_year(year: Optional[int] = None) -> datetime.date:
    if year is None:
        year = datetime.datetime.now().year
    return datetime.date(year, 12, 31)


def compute_gap_days_and_end_slot(slot: str, duration: str) -> Tuple[int, str]:
    """compute the end time given start time and duration.

    Args:
        slot (str): start intraday trading slot, i.e. 0931-1130 and 1301-1500.
        duration (str): duration (in trading time) string like '1d2h3m'.

    Returns:
        str: end intraday trading slot.
    """
    _d, _m = _parse_dhm(duration)
    slots = get_trading_slots(begin='0930', end="1500", freq_in_minute=1, bar_on_the_right=True)
    
    s_index = slots.index(slot)
    n_days, t_idnex = divmod(s_index + _m, len(slots))
    end = slots[t_idnex]
    n_days += _d
    return n_days, end
    

def load_tcalendar_list(
    begin: Optional[Datetime] = None, 
    end: Optional[Datetime] = None
) -> list[str]:
    """Load trading calendar from file.

    Returns:
        dict: trading calendar.
    """
    from pit.exceptions import CalendarMissingError
    cfg = read_config()
    if not os.path.exists(cfg.tcalendar_path):
        raise CalendarMissingError(f"Calendar {cfg.tcalendar_path} is missing")
    tc = pl.read_csv(cfg.tcalendar_path).get_column('date').to_list()
    
    _begin = any2ymd(begin) if begin else '1990-01-01'
    _end = any2ymd(end) if end else any2ymd(last_day_of_year())
    import bisect
    ll = bisect.bisect_left(tc, _begin)
    rr = bisect.bisect_right(tc, _end)
    tc = tc[ll:rr]
    return tc
    

def is_trading_day(date: Datetime) -> bool:
    """Check if a date is a trading day.

    Args:
        date (Datetime): date.

    Returns:
        bool: True if it is a trading day.
    """
    tc = load_tcalendar_list()
    _date = any2ymd(date)
    index = bisect.bisect_left(tc, _date)
    if index != len(tc) and tc[index] == _date:
        return True
    return False 


def get_tcalendar_df(n_next: int = 1) -> pl.DataFrame:
    """Get trading calendar dataframe.

    Args:
        n_next (int): number of next days.

    Returns:
        pl.DataFrame: trading calendar dataframe.
    """
    tcalendar = load_tcalendar_list()
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
    if n_shift == 0:
        raise ValueError("n_shift should not be 0, which is pointless.")
    _date = any2ymd(date)
    tcalendar_list = load_tcalendar_list()
    ii = bisect.bisect_left(tcalendar_list, _date)
    if tcalendar_list[ii] != _date and n_shift > 0:
        n_shift -= 1
    jj = ii + n_shift
    if jj < 0 or jj >= len(tcalendar_list):
        raise IndexError(f"Index out of calendar: {jj}")
    return datetime.datetime.strptime(tcalendar_list[jj], "%Y-%m-%d").date()


def adjust_tcalendar_slot_df(duration: str, start_slot: str = '0930') -> pl.DataFrame:
    """Adjust duration with shift.

    Args:
        duration (str): duration.
        start_slot (str): start slot.

    Returns:
        str: adjusted duration.
    """
    n_next, end_slot = compute_gap_days_and_end_slot(start_slot, duration)
    df = get_tcalendar_df(n_next)
    start_duration = pl.duration(hours=int(start_slot[:2]), minutes=int(start_slot[2:]), time_unit='ns')
    end_duration = pl.duration(hours=int(end_slot[:2]), minutes=int(end_slot[2:]), time_unit='ns')
    df = df.with_columns(
        pl.col('date').cast(pl.Datetime(time_unit='ns')).add(start_duration),
        pl.col('next').cast(pl.Datetime(time_unit='ns')).add(end_duration),
    )
    return df