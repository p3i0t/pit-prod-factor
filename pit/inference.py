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
from dlkit.utils import CHECHPOINT_META
from pit.datasource import (
    OfflineDataSource,
    OnlineV2DownsampleDataSource,
)

logger = logger.bind(where="inference")


__all__ = ["InferenceMode", "infer"]


class InferencePipeline:
    """The general inference pipeline for both offline and online inference."""

    def __init__(self, args: InferenceArguments):
        self.args = args
        self.models_available = sorted(os.listdir(f"{args.save_dir}/{args.prod}"))
        self.model_cache = {}

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
        self, infer_date: datetime.datetime, df_cs: pl.DataFrame
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
        ll = 0 if i - n_retain < 0 else i - n_retain
        models_required = self.models_available[ll:i]
        logger.info(
            f"infer on {infer_date:%Y-%m-%d}, models required: {models_required}"
        )

        # delete model cache that is not required, free GPU memory
        to_delete = [k for k in self.model_cache if k not in models_required]
        for k in to_delete:
            del self.model_cache[k]
            logger.info(f"delete model on {k}.")

        pred_list = []
        for model_date in models_required:
            ckpt_dir = f"{self.args.save_dir}/{self.args.prod}/{model_date}/{CHECHPOINT_META.prefix_dir}"
            if model_date in self.model_cache:
                _model, _normalizer = self.model_cache[model_date]
            else:
                _model, _normalizer = self._load_ckpt(ckpt_dir)
                self.model_cache[model_date] = (_model, _normalizer)

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


class InferenceMode(str, Enum):
    offline = "offline"
    online = "online"


def infer(
    args: InferenceArguments,
    infer_date: datetime.date | Tuple[datetime.date, datetime.date],
    *,
    mode: InferenceMode = InferenceMode.online,
    log_path: Optional[str] = None,
    verbose: bool = False,
) -> pl.DataFrame | Dict[int, pl.DataFrame] | None:
    log_path = log_path or "infer.log"
    logger.add(
        sink=f"{log_path}",
        filter=lambda record: record["extra"].get("where") == "inference",
        level="INFO",
    )

    infer_logger = logger.bind(where="inference")
    if isinstance(infer_date, datetime.date):
        begin, end = infer_date, infer_date
    elif isinstance(infer_date, tuple):
        begin, end = infer_date
    else:
        raise TypeError(
            f"date in InferenceArguments should be datetime.date"
            f"or Tuple[datetime.date, datetime.date], but is {type(infer_date)}"
        )

    infer_logger.info(f"Inference mode: {mode}, on prod: {args.prod}")
    if mode == InferenceMode.offline:
        ds = OfflineDataSource(
            data_path=str(args.dataset_dir) + "/*",
            columns=["date", "symbol"] + args.x_slot_columns,
            universe=args.universe,
            date_range=(begin, end),
            date_col="date",
            fill_nan=True,
        )
    elif mode == InferenceMode.online:
        # online data source.
        ds = OnlineV2DownsampleDataSource(
            slot_range=(args.x_begin, args.x_end),
            universe=args.universe,
            date_range=(begin, end),
            fill_nan=True,
            verbose=verbose,
        )
    else:
        raise ValueError(f"mode: {mode} not supported.")

    if verbose is True:
        s = perf_counter()
    df: pl.DataFrame = ds.collect()
    if verbose is True:
        df.write_parquet(f"df_{mode}.parq")

    if verbose is True:
        t = perf_counter() - s
        logger.info(f"time to load data: {t:.2f}s")
        s = perf_counter()
    ip = InferencePipeline(args=args)

    if verbose is True:
        t = perf_counter() - s
        logger.info(f"InferencePipeline pass time: {t:.2f}s")
    return ip(df)
