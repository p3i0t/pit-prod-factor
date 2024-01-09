from typing import List
import datetime
import itertools
import random

import pandas as pd
import numpy as np
import torch



def seed_everything(seed: int = 43):
    """set seed for reproducibility."""
    random.seed(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False



def generate_monthly_milestones(begin='2020-01-01', end='today', month_gap: int=1) -> list[str]:
    if end == 'today':
        end = datetime.datetime.today().strftime('%Y-%m-%d')

    year_begin = int(begin[:4])
    year_end = int(end[:4])
    years = list(range(year_begin, year_end + 1))

    milestones = [f"{y}-{m:02d}-01"
                  for y, m in itertools.product(years, range(1, 13, month_gap))]
    return [stone for stone in milestones if begin <= stone <= end]


def gen_rolling_date_splits(
    begin='2020-01-01',
    end='today',
    month_gap: int=1,
    trading_dates: list[str] = None,
    n_train: int = 800,
    n_valid: int=20,
    n_lag: int=5,
    min_train: int=500,
):
    milestones = generate_monthly_milestones(begin, end, month_gap=month_gap)
    milestones.append(trading_dates[-1])
    rolling_exp_splits = []
    for left_stone, right_stone in zip(milestones[:-1], milestones[1:]):
        trading_dates_earlier_than_milestone = []
        test_trading_dates = []
        for d in trading_dates:
            if d < left_stone:
                trading_dates_earlier_than_milestone.append(d)
            elif left_stone <= d < right_stone:
                test_trading_dates.append(d)
            else:
                break

        if len(trading_dates_earlier_than_milestone) <= n_valid + n_lag + min_train:
            continue
        train_keys = trading_dates_earlier_than_milestone[-(n_train+n_valid+n_lag):-(n_valid+n_lag)]
        valid_keys = trading_dates_earlier_than_milestone[-(n_valid+n_lag):-n_lag]
        lag_keys = trading_dates_earlier_than_milestone[-n_lag:]
        # if len(test_trading_dates) > 0:
        rolling_exp_splits.append((left_stone, (train_keys, valid_keys, lag_keys, test_trading_dates)))

    return rolling_exp_splits


def compute_ic(df: pd.DataFrame, x_cols: List[str], y_col: str):
    """Compute IC.
    Parameters
    ----------
    df : pd.DataFrame
        result pd.DataFrame contains columns including x_cols, y_col and ['symbol', 'date'].
    x_cols : List[str]
        list of names of left columns to compute IC.
    y_col : str
        name of right column to compute IC.
    Returns
    -------
    pd.DataFrame
        daily IC dataframe of given x_cols and y_col.
    """
    df_ = df[["date", "symbol", y_col] + x_cols]
    IC = df_.groupby("date").apply(lambda x: x[x_cols].corrwith(x[y_col]))
    return IC


def generate_slots(freq: int, bar_on_left: bool = True) -> list[tuple[str, str]]:
    """split the trading time into time slots given the `freq` as interval.

    Args:
        freq (str): munite interval to calculate derived columns, e.g. 5 or 10.
        bar_on_left (bool, optional): same as argument `bar_on_left` in
        http://docs.bopufund.com/toolset/docs/DataReader/read/stock.html. Defaults to True.
    """
    if bar_on_left:
        morning_s = datetime.datetime(2022, 1, 1, 9, 30)
        morning_t = morning_s + datetime.timedelta(minutes=freq - 1)
        afternoon_s = datetime.datetime(2022, 1, 1, 13, 0)
        afternoon_t = datetime.datetime(2022, 1, 1, 13, 0 + freq - 1)
    else:
        morning_s = datetime.datetime(2022, 1, 1, 9, 31)
        morning_t = morning_s + datetime.timedelta(minutes=freq-1)
        afternoon_s = datetime.datetime(2022, 1, 1, 13, 1)
        afternoon_t = datetime.datetime(2022, 1, 1, 13, 1 + freq - 1)

    interval = datetime.timedelta(seconds=60 * freq)

    morning_slots = []
    while morning_t.time() <= datetime.time(11, 30):
        _s = morning_s.time().strftime("%H%M")
        _t = morning_t.time().strftime("%H%M")
        morning_s += interval
        morning_t += interval

        morning_slots.append((_s, _t))

    afternoon_slots = []
    while afternoon_t.time() <= datetime.time(15, 0):
        _s = afternoon_s.time().strftime("%H%M")
        _t = afternoon_t.time().strftime("%H%M")
        afternoon_s += interval
        afternoon_t += interval

        afternoon_slots.append((_s, _t))

    return morning_slots + afternoon_slots

