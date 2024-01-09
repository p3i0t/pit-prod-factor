import os
import sys
import itertools
import datetime
import argparse
from functools import reduce
import yaml

import pandas as pd
import torch
import numpy as np

from ml_collections import config_dict
import genutils as gu

from utils import generate_slots
from utils.log import get_logger
from constants import FACTOR_ABBR

from model import XnY1Model



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='inference')
    parser.add_argument('--trading_slot', type=str, default='1030',
                        help='trading slot')
    parser.add_argument('--infer_date', type=str, default='today',
                        help='infer date')
    parser.add_argument('--n_latest', type=int, default=3,
                        help='use n model to ensemble.')
    args = parser.parse_args()
    cfg_dict = yaml.safe_load(open(f'config_{args.trading_slot}.yaml'))
    cfg = config_dict.ConfigDict(cfg_dict)
    # from pprint import pprint
    # pprint(cfg)

    log_dir = 'prod_log'
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    if args.infer_date == 'today':
        today: datetime.datetime = datetime.datetime.today()
    else:
        today: datetime.datetime = datetime.datetime.strptime(args.infer_date, '%Y-%m-%d')
    logger = get_logger(name=f'{today:%Y-%m-%d}_{args.trading_slot}', parent=log_dir)
    # dates = load_all_dates(data_dir=cfg.data_dir)
    # logger.info(f"{len(dates)} dates from {dates[0]} to {dates[-1]}")

    all_factors = [f"{b}_{agg}" for b, agg in itertools.product(FACTOR_ABBR, ['avg', 'stddevSamp'])]
    logger.info(f"input {len(all_factors)} factors over slots: {cfg.x_slots}.")

    x_cols = [_n + '_' + _t for _t, _n in itertools.product(cfg.x_slots, all_factors)]
    x_labels = np.array(x_cols).reshape((len(cfg.x_slots), len(all_factors)))
    y_cols = [f"rtn_{cfg.trading_slot}_{_d}D" for _d in cfg.returns]
    y_labels = np.array(y_cols)

    model_available_dates = sorted([path.split('.')[0] for path in os.listdir(cfg.checkpoint_dir)])
    logger.info(f"{model_available_dates=}")

    d_in = x_labels.shape[-1]
    d_out = len(cfg.returns)
    # model = XnY1Model(
    #     d_in=d_in,
    #     d_out=d_out,
    #     agg=cfg.agg,
    #     x_labels=x_labels,
    #     y_labels=y_labels,
    # )

    if len(gu.tcalendar.get(today, today)) == 0:
        logger.info('today is not a trading date.')
        sys.exit(0)

    if args.trading_slot == '0930':
        x_date = gu.tcalendar.adjust(today, -1)  # last trading date.
    else:
        x_date = today

    logger.info(f'use input data on {x_date}')
    agg_ops = ['avg', 'stddevSamp']
    agg_pairs = list(itertools.product(FACTOR_ABBR, agg_ops))


    import datareader as dr
    slots = generate_slots(freq=cfg.factor_freq, bar_on_left=False)
    slots_dict = {f'_{r}': (l, r) for l, r in slots if r in cfg.x_slots} # slots before `till`
    logger.info("start load online data")
    # fetch_max_limit = 100

    df_univ = dr.read(dr.meta.StockUniverse(['euniv_largemid']), begin=today, end=today)
    df_univ = df_univ[df_univ['euniv_largemid'].values]
    logger.info(f"universe `euniv_largemid` loaded, {len(df_univ)=}")
    df_list = []
    for k, v in slots_dict.items():
    # for i in range(0, len(agg_pairs), fetch_max_limit):
    #     _pairs = agg_pairs[i: i+fetch_max_limit]
        df = dr.read(
            dr.meta.StockMinuteDownsample(
                time={k: v}, columns=[], agg=agg_pairs,
                version='v2_5s', abbr=True, production=True),
            x_date, x_date)
        cols = df.columns.difference(['date', 'symbol'])
        # print(f"{len(cols)=}")
        logger.info(f"fetch slot {k} done, {len(cols)=}")
        df[cols] = df[cols].astype('float32')
        df_list.append(df)
    df = reduce(lambda x, y: x.merge(y, on=['date', 'symbol'], how='outer'), df_list)
    if args.trading_slot == '0930':
        df['date'] = df['date'].apply(lambda d: gu.tcalendar.adjust(d, 1))

    df = pd.merge(df_univ, df, on=['date', 'symbol'], how='left')
    print(df.head())

    logger.info(f"online data shape: {df.shape} ready.")

    today_str = today.strftime('%Y-%m-%d')
    # load models
    loaded_models_dict = {}
    model_dates = [d for d in model_available_dates if d <= today_str]
    if len(model_dates) == 0:
        raise Exception(f"No valid models for inference.")
    if len(model_dates) > args.n_latest:
        model_dates = model_dates[-args.n_latest:]

    logger.info(f"Inference on {today_str}, {model_dates=}")

    for model_date in model_dates:
        model = XnY1Model(
            d_in=d_in,
            d_out=d_out,
            agg=cfg.agg,
            x_labels=x_labels,
            y_labels=y_labels,
            device='cpu',
        )

        ckpt_path = f"{model_date}.pt"
        model.load_state_dict(torch.load(os.path.join(cfg.checkpoint_dir, ckpt_path)))
        # logger.info(f"{infer_date=}, load model from {latest_model_date}")
        loaded_models_dict[model_date] = model

    # _df = df[df.date == infer_date]
    x = df[list(x_labels.flatten())].to_numpy().reshape((len(df), *x_labels.shape))
    # print(f"{x=}")
    df_pred = df[['date', 'symbol']].copy()

    logger.info("daily batch data is ready")
    # mean of latest models
    pred_list = [model.predict_step(x) for k, model in loaded_models_dict.items()]
    df_pred[list(y_cols)] = sum(pred_list) / len(loaded_models_dict)
    print(df_pred)
    logger.info('inference done.')
    target_path = f"prediction_for_production/{today_str}_{cfg.trading_slot}.parq"
    df_pred.to_parquet(target_path)
    logger.info(f"predictions saved to {target_path}")


    df_alpha: pd.DataFrame = df_pred[['date', 'symbol', f'rtn_{cfg.trading_slot}_2D']].copy()
    df_alpha.rename(columns={f'rtn_{cfg.trading_slot}_2D': 'pit'}, inplace=True)
    df_alpha.set_index(['date', 'symbol'], inplace=True)

    alpha_path = f"/data/alpha/hub2/wangxin/prod_predictions/pit_{''.join(today_str.split('-'))}_{cfg.trading_slot[:2]}_{cfg.trading_slot[2:]}.parq"
    df_alpha.to_parquet(alpha_path)

    logger.info(f"alpha file for production: {alpha_path}")
