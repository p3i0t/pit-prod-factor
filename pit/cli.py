import click
from typing import Literal


# @click.command()
# @click.option('--source', default='bopu', help='source of calendar data')
# def update_calendar(source: Literal['akshare', 'bopu'] = 'bopu'):
#     import os
#     import pickle
#     script_directory = os.path.dirname(os.path.abspath(__file__))
#     if source == "akshare":
#         save_path = os.path.join(script_directory, 'ak_calendar.pkl')
#         import akshare as ak
#         df = ak.tool_trade_date_hist_sina()
#         trade_dates = [d.strftime('%Y-%m-%d') for d in sorted(df['trade_date']) if d.strftime('%Y-%m-%d') > '2017-01-01']
#         pickle.dump(trade_dates, open(save_path, 'wb'))
#     else:
#         try:
#             import genutils as gu  # noqa
#         except ImportError:
#             gu = None
#         trade_dates = gu.get_trade_dates()
#         save_path = os.path.join(script_directory, 'bopu_calendar.pkl')
#     return trade_dates

    