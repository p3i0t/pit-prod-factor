from typing import Optional
import itertools

import polars as pl


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


def downsample_1m_to_10m(
    df: pl.DataFrame | pl.LazyFrame, bars: Optional[list[str]] = None
) -> pl.DataFrame:
    assert "time" in df.columns, "column `time` not in df.columns"
    assert "date" in df.columns, "date not in df.columns"
    assert "symbol" in df.columns, "symbol not in df.columns"

    meta_cols = ["date", "symbol", "time"]
    if bars is None:
        bars = [_col for _col in df.columns if _col not in meta_cols]

    expr_list, agg_columns = bars_ops_combinations(bars, ops=['mean', 'std'])
    expr_list.append(pl.col("time").count().alias("count"))  # for debug

    if isinstance(df, pl.DataFrame):
        df = df.lazy()
    
    meta_cols = ["symbol", "date", "time"]
    # select proper input
    df = (
        df.select(meta_cols + bars)
        .sort(by=meta_cols)
        .filter(
            pl.col("time")
            .dt.strftime("%H%M")
            .is_between(pl.lit("0931"), pl.lit("1130"), closed="both")
            | pl.col("time")
            .dt.strftime("%H%M")
            .is_between(pl.lit("1301"), pl.lit("1500"), closed="both")
        )
    )
    _df = df.groupby_dynamic(
        index_column="time",
        every="10m",
        period="9m",
        offset="1m",
        by=["symbol", "date"],
        closed="both",
        include_boundaries=True,
    ).agg(expr_list).collect()

    _df: pl.DataFrame = _df.with_columns(
        [
            pl.col("_upper_boundary").dt.strftime("%H%M").alias("slot"),
            # pl.col('_upper_boundary').dt.strftime('%Y-%m-%d').alias('date')
        ]
    )

    slots = _df.get_column("slot").unique().sort().to_list()
    _df = _df.pivot(index=["symbol", "date"], columns="slot", values=agg_columns)
    name_mapping = {
        f"{col}_slot_{slt}": f"{col}_{slt}"
        for col, slt in itertools.product(agg_columns, slots)
    }
    _df = _df.rename(name_mapping)
    _df = _df.with_columns(pl.col(pl.NUMERIC_DTYPES).cast(pl.Float32))
    return _df