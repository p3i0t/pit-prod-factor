from typing import Union, Literal
from pathlib import Path
import polars as pl


def load_universe(glob_pattern: str, univ: str = None) -> pl.DataFrame:
    q: pl.LazyFrame = pl.scan_parquet(glob_pattern)
    if univ is not None:
        q = q.filter(pl.col(univ))
    q = q.select(['date', 'symbol']).sort(['date', 'symbol'])
    return q.collect()


def load_columns(glob_pattern: str, cols: list[str] = None) -> pl.DataFrame:
    # if cols is None:
    #     raise Warning(f"cols is None, only date, symbol will be loaded.")
    q: pl.LazyFrame = pl.scan_parquet(glob_pattern)
    if cols is not None:
        q = q.select(['date', 'symbol'] + cols)
    return q.collect()


def load_all_dates(
    data_dir: str,
    univ: str = 'euniv_largemid'
):
    with pl.StringCache():
        univ_pattern = str(Path(data_dir) / 'univ/*parquet')
        df: pl.DataFrame = load_universe(univ_pattern, univ=univ)
        df = df.with_columns(pl.col('date').dt.strftime('%Y-%m-%d'))
    all_dates = sorted(df.select(['date']).unique().to_numpy().squeeze())
    return all_dates


def load_df(
    data_dir: str,
    univ: str = 'euniv_largemid',
    dates: list[str] = None,
    return_dir: str = None,
    return_cols: Union [str, list[str]] = None,
    return_lag: Literal[0, 1]= 0,
    # barra_cols: Union[str, list[str]] = None,
    factor_dir: str = None,
    factor_cols: list[str] = None
) -> pl.DataFrame:
    with pl.StringCache():
        univ_pattern = str(Path(data_dir) / 'univ/*parquet')
        df_univ: pl.DataFrame = load_universe(univ_pattern, univ=univ)
        df_univ = df_univ.sort(by=['date',]) # fix data order, remove randomness

        # for r in return_cols:
        return_pattern = str(Path(data_dir) / f'{return_dir}/*parquet')
        df_return: pl.DataFrame = load_columns(return_pattern, cols=return_cols)
        if return_lag == 1:
            print('lag')
            date_lag_pattern = str(Path(data_dir) / 'date_lag/*parquet')
            df_lag: pl.DataFrame = pl.scan_parquet(date_lag_pattern).select(['date', 'prev']).collect()
            # df_lag: pl.DataFrame = load_columns(date_lag_pattern, cols=['date', 'prev'])
            print(df_return.head())
            df_return = df_return.join(df_lag, on='date', how='left')
            df_return = df_return.drop(columns=['date']).rename({'prev': 'date'})
            print(df_return.head())
        elif return_lag == 0:
            pass
        else:
            raise ValueError(f"{return_lag} must be 0 or 1.")

        df = df_univ.join(df_return, on=['date', 'symbol'], how='left')

        # barra_pattern = str(Path(data_dir) / 'barra/*parquet')
        # df_barra: pl.DataFrame = load_columns(barra_pattern, cols=barra_cols)
        # df = df.join(df_barra, on=['date', 'symbol'], how='left')

        factor_pattern = str(Path(data_dir) / f'{factor_dir}/*parquet')
        df_factor: pl.DataFrame = load_columns(factor_pattern, cols=factor_cols)
        df = df.join(df_factor, on=['date', 'symbol'], how='left')

        df = df.with_columns(pl.col('date').dt.strftime('%Y-%m-%d'))
        df = df.with_columns(pl.col('symbol').cast(pl.Utf8))

    df = df.filter(pl.col('date').is_in(dates))
    df = df.sort(by=['date', 'symbol'])
    return df
