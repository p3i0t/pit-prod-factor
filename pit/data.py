from typing import Union, Optional
import importlib
import polars as pl

from pit.configs import get_bars


class BopuDataReader:
    def __init__(self):
        try:
            self.dr = importlib.import_module("datareader")
        except ImportError:
            raise ImportError("Error: module datareader not found")
        try:
            self.gu = importlib.import_module("genutils")
        except ImportError:
            raise ImportError("Error: module genutils not found")

    def fetch_1min_bars(
        self, begin: str, end: str, cols: Optional[list[str]] = None
    ) -> pl.DataFrame:
        if cols is None:
            cols = get_bars('v3')
        df: pl.DataFrame = self.dr.read(
            self.dr.meta.StockMinute(columns=cols, version="3", abbr=True),
            begin=begin,
            end=end,
            df_lib='polars'
        )
        # df[cols] = df[cols].astype('float32')
        # df["symbol"] = df["symbol"].astype("category")
        # df = df.sort_values(by=['date', 'symbol']).reset_index(drop=True)
        return df
        # if len(df) >= 0:
        #     data_dir = Path(data_dir)
        #     data_dir.mkdir(parents=True, exist_ok=True)
        #     tgt_path =  data_dir / f'{date}.parq'
        #     df.to_parquet(tgt_path, index=False)
        # return date

    def fetch_univs(
        self, begin: str, end: str, univs: str | list[str] | None = None
    ) -> pl.DataFrame:
        if isinstance(univs, str):
            univs = [univs]
        elif univs is None:
            univs = [
                "univ_research",
                "univ_largemid",
                "sz50",
                "hs300",
                "zz500",
                "zz1000",
                "zz2000",
                "euniv_largemid",
                "euniv_research",
                "euniv_eresearch",
                "univ_full",
                "mktcap",
            ]
        else:
            univs = list(univs)

        df: pl.DataFrame = self.dr.read(
            self.dr.meta.StockUniverse(univs), begin=begin, end=end, df_lib='polars'
        )
        return df.select(["date", "symbol"] + univs).sort(by=["date", "symbol"])
        # df = df[["date", "symbol"] + univs]
        # df["symbol"] = df["symbol"].astype("category")
        # df = df.sort_values(by=["date", "symbol"]).reset_index(drop=True)
        # return df

    def fetch_returns(
        self, begin: str, end: str, n_list: int | list[int] | None = None
    ):
        if isinstance(n_list, tuple):
            n_list = list(n_list)
        elif isinstance(n_list, int):
            n_list = [n_list]
        elif n_list is None:
            n_list = [1, 2, 3, 5]

        # delay 5 minutes
        slots = [
            ("0935", "1000"),
            (1005, 1030),
            (1035, 1100),
            (1105, 1130),
            (1305, 1330),
            (1335, 1400),
            (1405, 1430),
            (1435, 1500),
        ]
        ret_slots = {}
        # vwap return
        for slot in slots:
            ret_slots[f"_v2v_{slot[0]}"] = (
                f"vwap_{slot[0]}_{slot[1]}",
                f"vwap_{slot[0]}_{slot[1]}",
            )
        # open-to-open return
        for slot in slots:
            ret_slots[f"_o2o_{slot[0]}"] = (f"close_{slot[0]}", f"close_{slot[0]}")

        df: pl.DataFrame = self.dr.read(
            self.dr.m.StockReturnDaily(
                ret_slots, n_days=n_list, abbr=False, future=True
            ),
            begin=begin,
            end=end,
            df_lib='polars'
        )
        
        # get intraday returns
        slots = [
            "0935",
            "1005",
            "1035",
            "1105",
            "1305",
            "1335",
            "1405",
            "1435",
            "1000",
            "1030",
            "1100",
            "1130",
            "1330",
            "1400",
            "1430",
            "1500",
        ]
        df_close: pl.DataFrame = self.dr.read(
            self.dr.m.StockMinute(["close"]),
            begin=begin,
            end=end,
            at=slots,
            df_lib='polars',
        )
        df_close = df_close.with_columns(pl.col('time').dt.strftime("%H%M").alias('slot'))
        # df_close["slot"] = df_close["time"].dt.strftime("%H%M")

        df_intra = df_close.pivot(
            values="close", columns="slot", index=["date", "symbol"]
        )
        df_intra = df_intra.with_columns(
            pl.col('1000').truediv(pl.col('0935')).sub(1).alias('0930_30m'),
            pl.col('1030').truediv(pl.col('1005')).sub(1).alias('1000_30m'),
            pl.col('1100').truediv(pl.col('1035')).sub(1).alias('1030_30m'),
            pl.col('1130').truediv(pl.col('1105')).sub(1).alias('1100_30m'),
            pl.col('1330').truediv(pl.col('1305')).sub(1).alias('1300_30m'),
            pl.col('1400').truediv(pl.col('1335')).sub(1).alias('1330_30m'),
            pl.col('1430').truediv(pl.col('1405')).sub(1).alias('1400_30m'),
            pl.col('1500').truediv(pl.col('1435')).sub(1).alias('1430_30m'),
            pl.col('1030').truediv(pl.col('0935')).sub(1).alias('0930_1h'),
            pl.col('1100').truediv(pl.col('1005')).sub(1).alias('1000_1h'),
            pl.col('1130').truediv(pl.col('1035')).sub(1).alias('1030_1h'),
            pl.col('1330').truediv(pl.col('1105')).sub(1).alias('1100_1h'),
            pl.col('1400').truediv(pl.col('1305')).sub(1).alias('1300_1h'),
            pl.col('1430').truediv(pl.col('1335')).sub(1).alias('1330_1h'),
            pl.col('1500').truediv(pl.col('1405')).sub(1).alias('1400_1h'),
        )
        # df_intra["0930_30m"] = df_intra["1000"] / df_intra["0935"] - 1
        # df_intra["1000_30m"] = df_intra["1030"] / df_intra["1005"] - 1
        # df_intra["1030_30m"] = df_intra["1100"] / df_intra["1035"] - 1
        # df_intra["1100_30m"] = df_intra["1130"] / df_intra["1105"] - 1
        # df_intra["1300_30m"] = df_intra["1330"] / df_intra["1305"] - 1
        # df_intra["1330_30m"] = df_intra["1400"] / df_intra["1335"] - 1
        # df_intra["1400_30m"] = df_intra["1430"] / df_intra["1405"] - 1
        # df_intra["1430_30m"] = df_intra["1500"] / df_intra["1435"] - 1

        # df_intra["0930_1h"] = df_intra["1030"] / df_intra["0935"] - 1
        # df_intra["1000_1h"] = df_intra["1100"] / df_intra["1005"] - 1
        # df_intra["1030_1h"] = df_intra["1130"] / df_intra["1035"] - 1
        # df_intra["1100_1h"] = df_intra["1330"] / df_intra["1105"] - 1
        # df_intra["1300_1h"] = df_intra["1400"] / df_intra["1305"] - 1
        # df_intra["1330_1h"] = df_intra["1430"] / df_intra["1335"] - 1
        # df_intra["1400_1h"] = df_intra["1500"] / df_intra["1405"] - 1

        intra_return_cols = [
            "0930_30m",
            "1000_30m",
            "1030_30m",
            "1100_30m",
            "1300_30m",
            "1330_30m",
            "1400_30m",
            "1430_30m",
            "0930_1h",
            "1000_1h",
            "1030_1h",
            "1100_1h",
            "1300_1h",
            "1330_1h",
            "1400_1h",
        ]

        df = df.join(
            df_intra.select(intra_return_cols + ["date", "symbol"]),
            on=["date", "symbol"],
            how="left",
        ).sort(by=["date", "symbol"])
        # df["symbol"] = df["symbol"].astype("category")
        # df = df.sort_values(by=["date", "symbol"]).reset_index(drop=True)
        return df

    def fetch_lag_returns(
        self, begin: str, end: str, n_list: int | list[int] | None = None
    ):
        df = self.fetch_returns(begin=begin, end=end, n_list=n_list)
        cols = ["date", "prev", "next"]
        date_lag = self.gu.tcalendar.getdf(begin=begin, end=end, renew=False)[cols]
        date_lag = pl.from_dataframe(date_lag)
        df_lag = df.join(date_lag, on="date", how="left")
        df_lag = df_lag.drop(["date", "next"])
        df_lag = df_lag.rename({"prev": "date"})
        cols_rename = {
            col: f"lag_{col}" for col in df.columns if col not in ["date", "symbol"]
        }
        df_lag = df_lag.rename(mapping=cols_rename)
        return df_lag