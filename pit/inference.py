import bisect
import datetime
import os
from collections import defaultdict
from copy import deepcopy
from enum import Enum
from time import perf_counter
from typing import Dict, Tuple, Optional

import numpy as np
import polars as pl
import torch
from loguru import logger

from dlkit.inference import InferenceArguments
from dlkit.models import get_model
from dlkit.preprocessing import DataFrameNormalizer
from dlkit.data import ParquetStockSource
from dlkit.utils import get_time_slots, CHECHPOINT_META
from pit.utils import normalize_date, process_offline_stock_df, Datetime
from pit import get_bars

logger = logger.bind(where="inference")


__all__ = ["InferenceDataSource", "infer"]

class InferencePipeline:
    """The general inference pipeline for both offline and online inference.
    """    
    def __init__(self, args: InferenceArguments):
        self.args = args
        self.models_available = sorted(os.listdir(f"{args.save_dir}/{args.prod}"))

    def _load_ckpt(self, ckpt_dir: str) -> Tuple[torch.nn.Module, DataFrameNormalizer]:
        """Load model and normalizer from checkpoint directory.

        Args:
            ckpt_dir (str): checkpoint directory.

        Returns:
            Tuple[torch.nn.Module, Normalizer]: model and normalizer.
        """

        model = get_model(
            name=self.args.model, d_in=self.args.d_in, d_out=self.args.d_out
        )
        model.load_state_dict(torch.load(f"{ckpt_dir}/model.pt"))
        normalizer = torch.load(f"{ckpt_dir}/normalizer.pt")
        return model, normalizer

    def __call__(
        self, df: pl.DataFrame, /
    ) -> Dict[int, pl.DataFrame] | pl.DataFrame | None:
        if df.shape[0] == 0:
            return pl.DataFrame()
        df_dict = df.partition_by("date", as_dict=True)  # dict of [date, df_cs]

        dict_list = []
        for date in sorted(df_dict.keys()):
            df_cs = df_dict[date]
            df_pred = self._forward_cross_sectional_batch(date, df_cs)
            dict_list.append(df_pred)

        merged_dict = defaultdict(list)
        for e in dict_list:
            for n, _df in e.items():
                merged_dict[n].append(_df)
        o = {n: pl.concat(df_list) for n, df_list in merged_dict.items()}
        if len(o) > 1:
            return o
        return list(o.values())[0]

    def _forward_cross_sectional_batch(
        self, infer_date: datetime.datetime = None, df_cs: pl.DataFrame = None
    ) -> Dict[int, pl.DataFrame] | None:
        """
        Inference on a cross-sectional batch of data,
        which is not supposed to be called directly.

        Args:
            infer_date: The date of the cross-sectional batch.
            df_cs: The cross-sectional batch of dataframe.

        Returns:
            prediction: The prediction dataframe.
        """
        if df_cs is None:
            raise ValueError("cross-sectional df must be provided.")
        if infer_date is None:
            raise ValueError("date must be provided.")
        df_cs = self._preprocess(df_cs)  # this is shared by all models
        df_index = df_cs.select(["date", "symbol"])

        # if isinstance(infer_date, datetime.datetime):
        _infer_date: str = infer_date.strftime("%Y-%m-%d")

        i = bisect.bisect_right(self.models_available, _infer_date)
        if i == 0:
            logger.error(f"no model available for date {infer_date}.")
            return None
        n_retain = (
            self.args.n_latest
            if isinstance(self.args.n_latest, int)
            else max(self.args.n_latest)
        )
        models_required = self.models_available[i - n_retain : i]
        logger.info(f"models required: {models_required}")

        pred_list = []
        for model_date in models_required:
            ckpt_dir = f"{self.args.save_dir}/{self.args.prod}/{model_date}/{CHECHPOINT_META.prefix_dir}"
            _model, _normalizer = self._load_ckpt(ckpt_dir)
            _df = _normalizer.transform(deepcopy(df_cs))  # this is model specific
            x = (
                _df.select(self.args.x_slot_columns)
                .to_numpy(order="c")
                .reshape(-1, *self.args.x_shape)
            )
            x = torch.Tensor(x).to(self.args.device, non_blocking=True)
            _model = _model.to(self.args.device)
            _model.eval()
            pred = _model(x)[:, -1, :]
            pred = pred.detach().cpu().numpy()
            pred_list.append(pred)

        n_latest = (
            [self.args.n_latest]
            if isinstance(self.args.n_latest, int)
            else self.args.n_latest
        )
        res_dict = {}
        for n in n_latest:
            pred = np.stack(pred_list[-n:]).mean(axis=0)
            df = df_index.hstack(pl.from_numpy(pred, schema=self.args.y_columns))
            res_dict[n] = df

        return res_dict

    def _preprocess(self, x: pl.DataFrame, /) -> pl.DataFrame:
        """preprocess the dataframe before feeding into the model.

        Args:
            x (pl.DataFrame): input cross-sectional dataframe.

        Returns:
            pl.DataFrame: _description_
        """
        cols = self.args.x_slot_columns
        x = x.with_columns(pl.col(c).fill_nan(pl.lit(None)) for c in cols)
        return x


class InferenceDataSource(str, Enum):
    offline = 'offline'
    online = 'online'


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


def load_and_process_online_stock_df(
    x_range: Tuple[str, str],
    universe: str,
    date_range: Optional[tuple[Datetime, Datetime]]=None,
) -> pl.DataFrame:
    import importlib
    import itertools
    import pandas as pd
    # from optimus.data.downsample import bars_ops_combinations

    cols = get_bars(feature_set='v2')
    try:
        dr = importlib.import_module("datareader")
    except ImportError:
        raise ImportError("Error: module datareader not found")

    x_begin, x_end = x_range
    if date_range:
        infer_begin, infer_end = date_range
    else:
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        infer_begin, infer_end = today, today
    
    online_logger = logger.bind(where="inference", function="online")
    # add universe filter
    slots_1min = get_time_slots(
        start=x_begin, end=x_end, freq_in_min=1
    )

    df_univ: pd.DataFrame = dr.read(
        dr.m.StockUniverse(universe),
        begin=infer_begin,
        end=infer_end,
    )

    s = perf_counter()
    df_1min: pd.DataFrame = dr.read(
        dr.meta.StockMinute(columns=cols, version="2", abbr=True, production=True),
        begin=infer_begin,
        end=infer_end,
        at=slots_1min,
    )
    t = perf_counter() - s
    online_logger.info(f"read 1min bars from datareader, time elapsed: {t:.2f}s")
    if df_1min.empty:
        online_logger.error("no 1min bars available.")
        # return None

    df_1min = df_univ.merge(df_1min, on=["date", "symbol"], how="left")

    # get full valid universe, i.e. symbols with complete 1min bars
    bar_cnt = df_1min[["symbol", "time"]].groupby("symbol").agg("count")
    bar_cnt = bar_cnt[bar_cnt["time"] == len(slots_1min)].reset_index()
    df_1min = df_1min[df_1min["symbol"].isin(bar_cnt["symbol"])]

    online_logger.info(f"valid 1min bars dataframe selected, shape: {df_1min.shape}")
    # online_logger.info(df_1min.head(20))
    # df_1min.to_parquet(f"{self.infer_date}.parq")

    s = perf_counter()
    meta_cols = ["date", "symbol", "time"]
    expr_list, agg_columns = bars_ops_combinations(cols, ops=["mean", "std"])
    expr_list.append(pl.col("time").count().alias("count"))  # for debug

    df = pl.from_pandas(df_1min)
    df = (
        df.lazy()
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
    online_logger.info(f"downsampling, time elapsed: {t:.2f}s")

    data_complete = all(df.select(pl.col("count") == 10).to_series().to_list())
    online_logger.info(f"all window has complete 10 minute bars: {data_complete}")

    # long to wide
    s = perf_counter()
    slots = df.get_column("slot").unique().sort().to_list()
    df = df.pivot(index=["symbol", "date"], columns="slot", values=agg_columns)
    name_mapping = {
        f"{col}_slot_{slt}": f"{col}_{slt}"
        for col, slt in itertools.product(agg_columns, slots)
    }
    df = df.rename(name_mapping)
    t = perf_counter() - s
    online_logger.info(
        f"dataframe pivot, shape: {df.shape}, time elapsed: {t:.2f}s"
    )
    return df

def infer(
    args: InferenceArguments,
    infer_date: str|Tuple[str, str] = 'today',
    mode: InferenceDataSource = InferenceDataSource.online,
    log_path: Optional[str] = None,
    debug: bool = False,
) -> pl.DataFrame | Dict[int, pl.DataFrame] | None:
    log_path = log_path or 'infer.log'
    logger.add(
        sink=f"{log_path}",
        filter=lambda record: record["extra"].get("where") == "inference",
        level="INFO",
        )
    
    infer_logger = logger.bind(where="inference")
    # construct data source
    if isinstance(infer_date, str):
        o = normalize_date(infer_date)
        begin, end = o, o
        infer_logger.info(f"{mode} inference on date: {o}")
    elif isinstance(infer_date, tuple):
        begin, end = infer_date
        begin = normalize_date(begin)
        end = normalize_date(end)
        infer_logger.info(f"{mode} inference on date range: {begin} to {end}")
    else:
        raise TypeError(f"date in InferenceArguments should be str or Tuple[str, str], but is {type(infer_date)}")

    date_col = 'date'
    if mode == InferenceDataSource.offline:
        ds_fn = ParquetStockSource(
            str(args.dataset_dir) + '/*',
            process_offline_stock_df,
            ["date", "symbol"] + args.x_slot_columns + args.y_columns,
            universe=args.universe,
            date_range=(begin, end),
            date_col=date_col,
            fill_nan=True,
            )
        with pl.StringCache():
            df = ds_fn()
    elif mode == InferenceDataSource.online:
        # online data source.
        df = load_and_process_online_stock_df(
            x_range=(args.x_begin, args.x_end),
            universe=args.universe,
            date_range=(begin, end),
        )
    else:
        raise ValueError(f"mode: {mode} not supported.")
    
    ip = InferencePipeline(args=args)
    if debug is True:
        df.write_csv(f"df_{mode}_{infer_date}.csv")
    return ip(df)
    