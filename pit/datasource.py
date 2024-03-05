from pathlib import Path
from typing import Optional, Protocol, Union, List, Tuple

from loguru import logger
import polars as pl

from dlkit.utils import get_time_slots
from pit import get_bars
from pit.utils import any2date, any2ymd, Datetime

__all__ = ["DataSource", "OfflineDataSource", "OnlineV2DownsampleDataSource", "Online10minDatareaderDataSource"]

class DataSource(Protocol):
    name = ""
    def collect(self) -> pl.DataFrame:
        ...

    
class OfflineDataSource(DataSource):
    """Data source loading offline data from disk.
    (1) select columns, e.g. `date`, `symbol`, x_columns, y_columns;
    (2) select samples by universe;
    (3) select samples in the date range;
    (4) fill NaN with null (i.e. None in polars); Unlike pandas, computation on NaN results in NaN in polars.

    Args:
        # df (pl.LazyFrame): input lazy polars dataframe.
        columns (list[str]): columns to select to construct dataset.
        universe (Optional[str], optional): universe to use. Defaults to None.
        date_range (Optional[tuple[Datetime, Datetime]], optional): filter on date range. Defaults to None.
        date_col (Optional[str], optional): date column for date range filtering. Defaults to None.
        fill_nan (bool, optional): whether fill NaN with null. Defaults to True.
    """
    name = "offline"
    def __init__(
        self, 
        data_path: Union[str, Path, List[str], List[Path]],
        columns: list[str],
        *,
        universe: Optional[str]=None, 
        date_range: Optional[tuple[Datetime, Datetime]]=None,
        date_col: Optional[str] = None,
        fill_nan: bool = True,
    ) -> None:
        super().__init__()
        df: pl.LazyFrame = pl.scan_parquet(data_path)

        if universe:
            df = df.filter(pl.col(universe))
        # must be behind universe filter, because `universe` column is not in `columns``.
        df = df.select(columns)  
        if date_range:
            begin, end = date_range
            begin = any2date(begin)
            end = any2date(end)
            if date_col is None:
                raise ValueError("date_col should not be None if date_range is not None.")
            else:
                if date_col not in columns:
                    raise ValueError(f"date_col {date_col} not available in columns.")
            df = df.filter(pl.col(date_col).is_between(pl.lit(begin), pl.lit(end)))
        if fill_nan:
            df = df.with_columns(pl.col(pl.NUMERIC_DTYPES).fill_nan(pl.lit(None)))
        self.df_lazy = df
            
    def collect(self) -> pl.DataFrame:
        return self.df_lazy.collect()
    

def bars_ops_combinations(
    bars: list[str], 
    ops: Optional[list[str]] = None
) -> tuple[list[pl.Expr], list[str]]:
    if ops is None:
        ops = ["mean", "std", "skew", "kurt"]
    expr_list = []
    columns = []
    for col in bars:
        for op in ops:
            if op == "mean":
                expr_list.append(pl.col(col).mean().alias(f"{col}_mean"))
                columns.append(f"{col}_mean")
            elif op == "std":
                expr_list.append(pl.col(col).std().alias(f"{col}_std"))
                columns.append(f"{col}_std")
            elif op == "skew":
                expr_list.append(pl.col(col).skew().alias(f"{col}_skew"))
                columns.append(f"{col}_skew")
            elif op == "kurt":
                expr_list.append(pl.col(col).kurtosis().alias(f"{col}_kurt"))
                columns.append(f"{col}_kurt")
    return expr_list, columns


class OnlineV2DownsampleDataSource(DataSource):
    """Data source loading online 1min data and downsampling using polars.
    """
    name = "online_downsample_v2"
    def __init__(
        self,
        slot_range: Tuple[str, str],
        *,
        universe: Optional[str] = None,
        date_range: Optional[tuple[Datetime, Datetime]]=None,
        fill_nan: bool = True,
        debug=False,
    ) -> None:
        super().__init__()
        self.slot_range = slot_range
        self.universe = universe
        self.date_range = date_range
        self.fill_nan = fill_nan
        self.debug = debug
        
            
    def collect(self) -> pl.DataFrame:
        import importlib
        import itertools
        from time import perf_counter

        cols = get_bars(feature_set='v2')
        try:
            dr = importlib.import_module("datareader")
        except ImportError:
            raise ImportError("Error: module datareader not found")

        x_begin, x_end = self.slot_range
        if self.date_range:
            infer_begin, infer_end = self.date_range
        else:
            today = any2ymd("today")
            infer_begin, infer_end = today, today
        
        slots_1min = get_time_slots(
            start=x_begin, end=x_end, freq_in_min=1
        )

        s = perf_counter()
        df_1min: pl.DataFrame = dr.read(
            dr.meta.StockMinute(columns=cols, version="2", abbr=True, production=True),
            begin=infer_begin,
            end=infer_end,
            at=slots_1min,
            df_lib='polars'
        )
        t = perf_counter() - s
        logger.info(f"read 1min bars from datareader, time elapsed: {t:.2f}s")
        
        if self.universe:
            df_univ: pl.DataFrame = dr.read(
                dr.m.StockUniverse(self.universe),
                begin=infer_begin,
                end=infer_end,
                df_lib='polars'
            )
            df_1min = df_univ.join(df_1min, on=["date", "symbol"], how="left")

        if df_1min.shape[0] == 0:
            logger.error("no 1min bars available.")

        # get full valid universe, i.e. symbols with complete 1min bars
        symbols_complete: pl.Series = (
            df_1min
            .select(['symbol', 'time']).group_by('symbol')
            .agg(pl.col('time').count().alias('count'))
            .filter(pl.col('count') == len(slots_1min))
            .get_column('symbol')
            )
        df_1min = df_1min.filter(pl.col('symbol').is_in(symbols_complete))

        s = perf_counter()
        meta_cols = ["date", "symbol", "time"]
        expr_list, agg_columns = bars_ops_combinations(cols, ops=["mean", "std"])
        expr_list.append(pl.col("time").count().alias("count"))  # for debug

        df = (
            df_1min.lazy()
            .sort(by=meta_cols)
            .group_by_dynamic(
                index_column="time",
                every="10m",
                period="9m",
                offset="1m",
                by=["symbol", "date"],
                closed="both",
                include_boundaries=True,
            )
            .agg(expr_list)
        ).collect()

        df: pl.DataFrame = df.with_columns(
            [
                pl.col("_upper_boundary").dt.strftime("%H%M").alias("slot"),
                # pl.col('_upper_boundary').dt.strftime('%Y-%m-%d').alias('date')
            ]
        )
        t = perf_counter() - s
        logger.info(f"downsampling, time elapsed: {t:.2f}s")
        data_complete = all(df.select(pl.col("count") == 10).to_series().to_list())
        logger.info(f"all window has complete 10 minute bars: {data_complete}")

        # long to wide
        s = perf_counter()
        slots = df.get_column("slot").unique().sort().to_list()
        if self.debug is True:
            logger.info("[debug] df after downsample:", df.select(["date", "symbol"]).head(5))
        df = df.pivot(index=["symbol", "date"], columns="slot", values=agg_columns)
        if self.debug is True:
            logger.info("[debug] df after pivot:", df.select(["date", "symbol"]).head(5))
        name_mapping = {
            f"{col}_slot_{slt}": f"{col}_{slt}"
            for col, slt in itertools.product(agg_columns, slots)
        }
        df = df.rename(name_mapping)

        if self.fill_nan:
            df = df.with_columns(pl.col(pl.NUMERIC_DTYPES).fill_nan(pl.lit(None)))
        t = perf_counter() - s
        logger.info(
            f"dataframe pivot, shape: {df.shape}, time elapsed: {t:.2f}s"
        )
        return df


class Online10minDatareaderDataSource(DataSource):
    """Data source for inference loading 10min downsampled data from datareader.
    """
    name = "online_10min_datareader"
    def __init__(
        self, 
        slot_range: Tuple[str, str],
        *,
        universe: Optional[str] = None,
        date_range: Optional[tuple[Datetime, Datetime]]=None,
        fill_nan: bool = True,
    ) -> None:
        super().__init__()
        self.slot_range = slot_range
        self.universe = universe
        self.date_range = date_range
        self.fill_nan = fill_nan

    def collect(self) -> pl.DataFrame:
        import importlib

        cols = get_bars(feature_set='v2')
        try:
            dr = importlib.import_module("datareader")
        except ImportError:
            raise ImportError("Error: module datareader not found")
        x_slots_l = get_time_slots(
            start=self.slot_range[0], end=self.slot_range[1], freq_in_min=10, bar_on_the_right=False
        )
        x_slots_r = get_time_slots(
            start=self.slot_range[0], end=self.slot_range[1], freq_in_min=10, bar_on_the_right=True
        )
        agg_pairs = ['avg', 'stddevSamp']
        slot_dict = {r: (l, r) for l, r in zip(x_slots_l, x_slots_r)}

        df_merged = pl.DataFrame()        
        for r_slot, _slot_range in slot_dict.items():
            # for loop in case the datareader explodes.
            begin, end = _slot_range
            df: pl.DataFrame = dr.read(
                dr.meta.StockMinuteDownsample(
                    time={r_slot: _slot_range}, 
                    columns=cols, 
                    agg=agg_pairs, 
                    version="2", abbr=True, 
                    production=True),
                begin=begin, end=end, df_lib='polars'
            )
            if df_merged.is_empty():
                df_merged = df
            else:
                df_merged = df_merged.join(df, on=["date", "symbol"], how="left")
        
        if self.universe:
            df_univ: pl.DataFrame = dr.read(
                dr.m.StockUniverse(self.universe),
                begin=begin,
                end=end,
                df_lib='polars'
            )
            df_merged = df_univ.join(df_merged, on=["date", "symbol"], how="left")
            
        df_merged.columns = [c.replace('avg', 'mean').replace('stddevSamp', 'std') for c in df_merged.columns]
        if self.fill_nan:
            df_merged = df_merged.with_columns(pl.col(pl.NUMERIC_DTYPES).fill_nan(pl.lit(None)))
        return df_merged