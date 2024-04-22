from datetime import datetime
from time import perf_counter
from typing import Tuple, Optional

from loguru import logger
import polars as pl
import torch
from dlkit.preprocessing import DataFrameNormalizer, StandardScaler, CrossSectionalScaler
from dlkit.train import StockTrainer, TrainArguments
from dlkit.data import StockDataset
from dlkit.utils import CHECHPOINT_META

from pit.datasource import OfflineDataSource

train_logger = logger.bind(where="train_pipeline")


__all__ = ["TrainPipeline"]

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
        self.offline_ds = OfflineDataSource(
            data_path=str(self.args.dataset_dir) + '/*',
            columns=["date", "symbol"] + self.args.x_slot_columns + self.args.y_columns,
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
            df = self.offline_ds.collect()
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
        logger.add(sink=f"{self.args.milestone_dir}/train.log", level="INFO")
        logger.info(f"start training pipeline on {self.args.milestone}.")
        train_set, eval_set, test_set = self.split_data()
        
        # normalize
        train_set = self.normalizer.fit_transform(train_set)
        eval_set = self.normalizer.transform(eval_set)
        if test_set is not None:
            test_set = self.normalizer.transform(test_set)
        train_logger.info("normalization done.")
        
        train_set = self.df_to_dataset(train_set)
        eval_set = self.df_to_dataset(eval_set)
        if test_set is not None:
            test_set = self.df_to_dataset(test_set)
        
        if train_set and eval_set:
            trainer = StockTrainer(
                args=self.args,
                train_dataset=train_set,
                eval_dataset=eval_set,
            )
            train_logger.info("start training.")
            trainer.train()
            train_logger.info("training done.")
            checkpoint_dir = f"{self.args.milestone_dir}/{CHECHPOINT_META.prefix_dir}"
            torch.save(self.normalizer, f"{checkpoint_dir}/{CHECHPOINT_META.normalizer}")
        else:
            raise ValueError("train_set and eval_set must not be None.")
