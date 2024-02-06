from datetime import datetime
from time import perf_counter
from typing import Tuple, Optional

from loguru import logger
import polars as pl
import torch
from dlkit.preprocessing import DataFrameNormalizer, StandardScaler, CrossSectionalScaler
from dlkit.train import StockTrainer, TrainArguments
from dlkit.data import StockDataset, ParquetStockSource
from dlkit.utils import CHECHPOINT_META

from pit.utils import Datetime, normalize_date

train_logger = logger.bind(where="train_pipeline")


def process_stock_df(
    df: pl.LazyFrame, 
    columns: list[str], 
    *,
    universe: Optional[str]=None, 
    date_range: Optional[tuple[Datetime, Datetime]]=None,
    date_col: Optional[str] = None,
    fill_nan: bool = True,
    ) -> pl.LazyFrame:
    """Preprocess on a polars.LazyFrame, include: 
    (1) select columns, e.g. `date`, `symbol`, x_columns, y_columns;
    (2) select samples by universe;
    (3) select samples in the date range;
    (4) fill NaN with null (i.e. None in polars); Unlike pandas, computation on NaN results in NaN in polars.

    Args:
        df (pl.LazyFrame): input lazy polars dataframe.
        columns (list[str]): columns to select to construct dataset.
        universe (Optional[str], optional): universe to use. Defaults to None.
        date_range (Optional[tuple[Datetime, Datetime]], optional): filter on date range. Defaults to None.
        date_col (Optional[str], optional): date column for date range filtering. Defaults to None.
        fill_nan (bool, optional): whether fill NaN with null. Defaults to True.

    Raises:
        ValueError: _description_
        ValueError: _description_

    Returns:
        pl.LazyFrame: preprocessed polars.LazyFrame.
    """
    if universe:
        df = df.filter(pl.col(universe))
    # must be behind universe filter, because `universe` column is not in `columns``.
    df = df.select(columns)  
    if date_range:
        begin, end = date_range
        begin = normalize_date(begin)
        end = normalize_date(end)
        if date_col is None:
            raise ValueError("date_col should not be None if date_range is not None.")
        else:
            if date_col not in columns:
                raise ValueError(f"date_col {date_col} not available in columns.")
            
        df = df.filter(pl.col(date_col).is_between(pl.lit(begin), pl.lit(end)))
    if fill_nan:
        df = df.with_columns(pl.col(pl.NUMERIC_DTYPES).fill_nan(pl.lit(None)))
    df = df.with_columns(
        pl.col('date').cast(pl.Utf8),
        pl.col('symbol').cast(pl.Utf8),
        )
    return df   


def get_normalizer(name: str = "zscore") -> DataFrameNormalizer:
    if name == "zscore":
        return StandardScaler()
    elif name == 'cs_zscore':
        return CrossSectionalScaler()
    else:
        raise ValueError(f"Unknown normalizer: {name}")


class TrainPipeline:
    def __init__(self, args: TrainArguments, debug: bool = False):
        self.args = args
        # construct data source
        begin = self.args.train_date_range[0]
        if self.args.test_date_range:
            end = self.args.test_date_range[-1] 
        else:
            end = self.args.eval_date_range[-1] 
        self.date_col = 'date'
        self.data_source = ParquetStockSource(
            str(self.args.dataset_dir) + '/*',
            process_stock_df,
            ["date", "symbol"] + self.args.x_slot_columns + self.args.y_columns,
            universe=self.args.universe,
            date_range=(begin, end),
            date_col=self.date_col,
            fill_nan=True,
        )
        self.normalizer = get_normalizer(name=args.normalizer)
        self.debug = debug

    
    def split_data(self) -> Tuple[pl.DataFrame, pl.DataFrame, Optional[pl.DataFrame]]:
        with pl.StringCache():
            s = perf_counter()
            df = self.data_source()
            t = perf_counter() - s
            train_logger.info(f"time to fetch data: {t:.2f}s")

            train_begin, train_end = self.args.train_date_range
            train_set = df.filter(
                pl.col(self.date_col).is_between(
                    pl.lit(datetime.strptime(train_begin, "%Y-%m-%d")),
                    pl.lit(datetime.strptime(train_end, "%Y-%m-%d")),
                )
            )
            train_logger.info(
                f"{len(train_set)} train samples from {train_begin} to {train_end}."
            )

            eval_begin, eval_end = self.args.eval_date_range
            eval_set = df.filter(
                pl.col(self.date_col).is_between(
                    pl.lit(datetime.strptime(eval_begin, "%Y-%m-%d")),
                    pl.lit(datetime.strptime(eval_end, "%Y-%m-%d")),
                )
            )
            train_logger.info(
                f"{len(eval_set)} eval samples from {eval_begin} to {eval_end}."
            )

            test_set = None
            if self.args.test_date_range is not None:
                test_begin, test_end = self.args.test_date_range
                test_set = df.filter(
                    pl.col(self.date_col).is_between(
                        pl.lit(datetime.strptime(test_begin, "%Y-%m-%d")),
                        pl.lit(datetime.strptime(test_end, "%Y-%m-%d")),
                    )
                )
                train_logger.info(
                    f"{len(test_set)} test samples from {test_begin} to {test_end}."
                )
            return train_set, eval_set, test_set

    def df_to_dataset(self, df: Optional[pl.DataFrame] = None) -> Optional[StockDataset]:
        if df is None:
            return None
        return StockDataset(
            date=df.get_column("date").to_numpy(),
            symbol=df.get_column("symbol").to_numpy(),
            x=df.select(self.args.x_slot_columns)
            .to_numpy(order="c")
            .reshape(-1, *self.args.x_shape),
            y=df.select(self.args.y_columns)
            .to_numpy(order="c")
            .reshape(-1, *self.args.y_shape),
            y_columns=self.args.y_columns,
        )

    def run(self):
        train_set, eval_set, test_set = self.split_data()
        
        # normalize
        train_set = self.normalizer.fit_transform(train_set)
        eval_set = self.normalizer.transform(eval_set)
        if test_set is not None:
            test_set = self.normalizer.transform(test_set)
        train_logger.info("normalization done.")
        
        train_set = self.df_to_dataset(train_set)
        if self.debug:
            train_logger.info(f"{train_set.date.shape=}, {type(train_set.date)}")
            train_logger.info(f"{train_set.symbol.shape=}, {type(train_set.symbol)}")
            train_logger.info(f"{train_set.x.shape=}, {type(train_set.x)}")
            train_logger.info(f"{train_set.y.shape=}, {type(train_set.y)}")
        eval_set = self.df_to_dataset(eval_set)
        if test_set is not None:
            test_set = self.df_to_dataset(test_set)
        
        if train_set and eval_set:
            trainer = StockTrainer(
                args=self.args,
                train_dataset=train_set,
                eval_dataset=eval_set,
            )
            if self.debug:
                train_logger.info("before train()")
            trainer.train()
            checkpoint_dir = f"{self.args.milestone_dir}/{CHECHPOINT_META.prefix_dir}"
            torch.save(self.normalizer, f"{checkpoint_dir}/{CHECHPOINT_META.normalizer}")
        else:
            raise ValueError("train_set and eval_set must not be None.")
