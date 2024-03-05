from typing import TypeVar, Annotated
import datetime

import dateutil.parser

__all__ = ['any2datetime', 'any2date', 'any2ymd']

Datetime = TypeVar('Datetime', 
                   Annotated[str, "string like '20220201', '2022-02-01' or 'today'."], 
                   datetime.datetime, 
                   Annotated[int, "int like 20220201."])

def any2datetime(dt: Datetime) -> datetime.datetime:
    if isinstance(dt, datetime.datetime):
        return dt
    elif isinstance(dt, datetime.date):
        return datetime.datetime.combine(dt, datetime.time())
    if isinstance(dt, int):
        return dateutil.parser.parse(str(dt))
    if isinstance(dt, str):
        if dt == 'today':
            o = datetime.datetime.combine(
                    datetime.datetime.now().date(), datetime.time()
                )
        else:
            o = dateutil.parser.parse(dt)
        return o
    else:
        raise TypeError(f"dt type {type(dt)} not supported.")

def any2date(ts_input) -> datetime.date:
    return any2datetime(ts_input).date()

def any2ymd(ts_input) -> str:
    return any2date(ts_input).strftime("%Y-%m-%d")

# def any2time(ts_input) -> datetime.time:
#     return any2datetime(ts_input).time()

# def any2hm(ts_input) -> str:

# def process_offline_stock_df(
#     df: pl.LazyFrame, 
#     columns: list[str], 
#     *,
#     universe: Optional[str]=None, 
#     date_range: Optional[tuple[Datetime, Datetime]]=None,
#     date_col: Optional[str] = None,
#     fill_nan: bool = True,
#     ) -> pl.LazyFrame:
#     """Preprocess on a polars.LazyFrame, include: 
#     (1) select columns, e.g. `date`, `symbol`, x_columns, y_columns;
#     (2) select samples by universe;
#     (3) select samples in the date range;
#     (4) fill NaN with null (i.e. None in polars); Unlike pandas, computation on NaN results in NaN in polars.

#     Args:
#         df (pl.LazyFrame): input lazy polars dataframe.
#         columns (list[str]): columns to select to construct dataset.
#         universe (Optional[str], optional): universe to use. Defaults to None.
#         date_range (Optional[tuple[Datetime, Datetime]], optional): filter on date range. Defaults to None.
#         date_col (Optional[str], optional): date column for date range filtering. Defaults to None.
#         fill_nan (bool, optional): whether fill NaN with null. Defaults to True.

#     Raises:
#         ValueError: _description_
#         ValueError: _description_

#     Returns:
#         pl.LazyFrame: preprocessed polars.LazyFrame.
#     """
#     if universe:
#         df = df.filter(pl.col(universe))
#     # must be behind universe filter, because `universe` column is not in `columns``.
#     df = df.select(columns)  
#     if date_range:
#         begin, end = date_range
#         begin = any2date(begin)
#         end = any2date(end)
#         if date_col is None:
#             raise ValueError("date_col should not be None if date_range is not None.")
#         else:
#             if date_col not in columns:
#                 raise ValueError(f"date_col {date_col} not available in columns.")
#         df = df.filter(pl.col(date_col).is_between(pl.lit(begin), pl.lit(end)))
#     if fill_nan:
#         df = df.with_columns(pl.col(pl.NUMERIC_DTYPES).fill_nan(pl.lit(None)))
#     # df = df.with_columns(
#     #     # pl.col('date').cast(pl.Utf8),
#     #     pl.col('symbol').cast(pl.Utf8),
#     #     )
#     return df   