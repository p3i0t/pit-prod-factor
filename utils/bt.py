from pathlib import Path
import pandas as pd
import datareader as dr
from factoranalyzer import FactorAnalyzer, PortfolioConstructor
from backtester import BackTester #parallel_load
from backtester.core.config import get_yaml


def call_fa(df: pd.DataFrame, method='top_bottom', slot: str = None, template: str = None, limit: bool=True):
    # from idmapper import LOG as IMLOG
    # IMLOG.handlers[0].setLevel("WARNING")

    # dr.config.set_logging_level('CRITICAL')
    # print(f"pre dr log level: {dr.LOG.handlers[0].level=}")
    # print(f"post dr log level: {dr.LOG.handlers[0].level=}")
    # h, m = slot.split('_')
    trading_start = f"{slot[:2]}:{slot[2:]}"
    # print(f"reports_{today}/report_{slot}_{method}_short_universe.pdf")
    analyzer = FactorAnalyzer()
    analyzer.set_factors(df, universe='univ_largemid')
    # analyzer.construct_portfolios(buckets=20)
    # if method == 'top_bottom':
    #     mm = PortfolioConstructor.TopBottom()
    # elif method == 'standard_winsor':
    #     mm = PortfolioConstructor.StandardWinsor()

    # analyzer.construct_portfolios(buckets=10)
    analyzer.construct_portfolios(buckets=10, universe_avg_as_short=True)
    # analyzer.construct_portfolios(method=long_only=)
    if limit:
        analyzer.backtest(
            commission_buy=0,
            commission_sell=0,
            # commission_buy=2e-4,
            # commission_sell=12e-4,
            default_trading_start=trading_start,
            missing_target='clear',
            ignore_all_limit=False,
            limit_down=True,
            limit_up=True,
            limit_bothway=True
        )
    else:
        analyzer.backtest(
            commission_buy=0,
            commission_sell=0,
            # commission_buy=2e-4,
            # commission_sell=12e-4,
            default_trading_start=trading_start,
            missing_target='clear',
            ignore_all_limit=True,
            limit_down=False,
            limit_up=False,
            limit_bothway=False
        )
    # analyzer.report(f"reports_{today}/report_{slot}_{train_type}.pdf")
    analyzer.report(f"{template}.pdf", )
