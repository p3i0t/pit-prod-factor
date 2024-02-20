from enum import Enum
import os
from typing import Optional

from dlkit.train import TrainArguments
from dlkit.inference import InferenceArguments


from omegaconf import OmegaConf, DictConfig
# OmegaConf.register_new_resolver(
#     "resolve_returns", 
#     lambda prefix, slot, delay, durations, suffix: 
#         [f"{prefix}{int(slot)+int(delay):04d}_{d}{suffix}" for d in durations]
#     )
# OmegaConf.register_new_resolver(
#     "resolve_d_out", lambda durations: len(durations)
# )
# OmegaConf.register_new_resolver(
#     'resolve_x_slots', lambda x_begin, x_end, x_freq: get_time_slots(x_begin, x_end, freq_in_min=x_freq)
# )
 
 
__all__ = ['get_bars', 'get_training_config', 'get_inference_config', 'list_prods', 'ProdsAvailable']

bar_v2 = [
    "data_source",
    "arrival_time",
    "open",
    "close",
    "high",
    "low",
    "high_to_now",
    "low_to_now",
    "accvolume",
    "volume",
    "accamount",
    "amount",
    "open_amount",
    "close_amount",
    "high_amount",
    "low_amount",
    "iopv",
    "total_trades",
    "acc_total_trades",
    "open_ask1_price",
    "open_ask1_size",
    "open_bid1_price",
    "open_bid1_size",
    "close_ask1_price",
    "close_ask1_size",
    "close_bid1_price",
    "close_bid1_size",
    "high_ask1_price",
    "high_ask1_size",
    "high_bid1_price",
    "high_bid1_size",
    "low_ask1_price",
    "low_ask1_size",
    "low_bid1_price",
    "low_bid1_size",
    "avg_ask1_price",
    "avg_ask1_size",
    "avg_bid1_price",
    "avg_bid1_size",
    "vwap_ask1_price",
    "vwap_bid1_price",
    "open_mid_price",
    "close_mid_price",
    "mid_price_avg",
    "mid_price_std",
    "mid_price_skew",
    "mid_price_kurt",
    "min_spread",
    "max_spread",
    "avg_spread",
    "open_ask_amount10",
    "open_bid_amount10",
    "close_ask_amount10",
    "close_bid_amount10",
    "avg_ask_amount10",
    "avg_bid_amount10",
    "ask_volume10_avg",
    "bid_volume10_avg",
    "open_vwap_ask_price10",
    "open_avg_ask_price10",
    "open_vwap_bid_price10",
    "open_avg_bid_price10",
    "close_vwap_ask_price10",
    "close_avg_ask_price10",
    "close_vwap_bid_price10",
    "close_avg_bid_price10",
    "vwap_ask_price10_avg",
    "avg_ask_price10_avg",
    "vwap_bid_price10_avg",
    "avg_bid_price10_avg",
    "delta_amount_ask_algo1",
    "delta_amount_bid_algo1",
    "delta_amount_ask_algo2",
    "delta_amount_bid_algo2",
    "delta_amount_ask_algo3",
    "delta_amount_bid_algo3",
    "delta_amount_ask_algo4",
    "delta_amount_bid_algo4",
    "qimb1_avg",
    "qimb1_std",
    "qimb1_skew",
    "qimb1_kurt",
    "qimb10_avg",
    "qimb10_std",
    "qimb10_skew",
    "qimb10_kurt",
    "tick_return_avg",
    "tick_return_std",
    "tick_return_skew",
    "tick_return_kurt",
    "ask_amount10_chg_avg",
    "ask_amount10_chg_std",
    "ask_amount10_chg_skew",
    "ask_amount10_chg_kurt",
    "bid_amount10_chg_avg",
    "bid_amount10_chg_std",
    "bid_amount10_chg_skew",
    "bid_amount10_chg_kurt",
    "ask_amount10_ratio1_avg",
    "ask_amount10_ratio1_std",
    "ask_amount10_ratio1_skew",
    "ask_amount10_ratio1_kurt",
    "bid_amount10_ratio1_avg",
    "bid_amount10_ratio1_std",
    "bid_amount10_ratio1_skew",
    "bid_amount10_ratio1_kurt",
    "ask_amount10_ratio2_avg",
    "ask_amount10_ratio2_std",
    "ask_amount10_ratio2_skew",
    "ask_amount10_ratio2_kurt",
    "bid_amount10_ratio2_avg",
    "bid_amount10_ratio2_std",
    "bid_amount10_ratio2_skew",
    "bid_amount10_ratio2_kurt",
    "book10_ratio_avg",
    "book10_ratio_std",
    "book10_ratio_skew",
    "book10_ratio_kurt",
    "book10_ratio_chg_avg",
    "book10_ratio_chg_std",
    "book10_ratio_chg_skew",
    "book10_ratio_chg_kurt",
    "book10_rratio_avg",
    "book10_rratio_std",
    "book10_rratio_skew",
    "book10_rratio_kurt",
    "book10_rratio_chg_avg",
    "book10_rratio_chg_std",
    "book10_rratio_chg_skew",
    "book10_rratio_chg_kurt",
    "twap",
    "arrival_time_from_trans",
    "total_trades_from_trans",
    "twap_from_trans",
    "buy_amount_by_bsflag_from_trans",
    "sell_amount_by_bsflag_from_trans",
    "buy_amount_by_tick_from_trans",
    "sell_amount_by_tick_from_trans",
    "buy_amount_by_quote_from_trans",
    "sell_amount_by_quote_from_trans",
    "vwap",
    "qimb_1_close",
    "qimb_10",
    "timb_bsflag",
    "timb_tick",
    "timb_quote",
    "mpb",
    "tpb",
    "hml",
]
bar_v3 = [
    "data_source",
    "arrival_time",
    "open",
    "close",
    "high",
    "low",
    "high_to_now",
    "low_to_now",
    "accvolume",
    "volume",
    "accamount",
    "amount",
    "open_amount",
    "close_amount",
    "high_amount",
    "low_amount",
    "iopv",
    "total_trades",
    "acc_total_trades",
    "open_ask1_price",
    "open_ask1_size",
    "open_bid1_price",
    "open_bid1_size",
    "close_ask1_price",
    "close_ask1_size",
    "close_bid1_price",
    "close_bid1_size",
    "high_ask1_price",
    "high_ask1_size",
    "high_bid1_price",
    "high_bid1_size",
    "low_ask1_price",
    "low_ask1_size",
    "low_bid1_price",
    "low_bid1_size",
    "avg_ask1_price",
    "avg_ask1_size",
    "avg_bid1_price",
    "avg_bid1_size",
    "vwap_ask1_price",
    "vwap_bid1_price",
    "open_mid_price",
    "close_mid_price",
    "mid_price_avg",
    "mid_price_std",
    "mid_price_skew",
    "mid_price_kurt",
    "min_spread",
    "max_spread",
    "avg_spread",
    "open_ask_amount10",
    "open_bid_amount10",
    "close_ask_amount10",
    "close_bid_amount10",
    "avg_ask_amount10",
    "avg_bid_amount10",
    "ask_volume10_avg",
    "bid_volume10_avg",
    "open_vwap_ask_price10",
    "open_avg_ask_price10",
    "open_vwap_bid_price10",
    "open_avg_bid_price10",
    "close_vwap_ask_price10",
    "close_avg_ask_price10",
    "close_vwap_bid_price10",
    "close_avg_bid_price10",
    "vwap_ask_price10_avg",
    "avg_ask_price10_avg",
    "vwap_bid_price10_avg",
    "avg_bid_price10_avg",
    "delta_amount_ask_algo1",
    "delta_amount_bid_algo1",
    "delta_amount_ask_algo2",
    "delta_amount_bid_algo2",
    "delta_amount_ask_algo3",
    "delta_amount_bid_algo3",
    "delta_amount_ask_algo4",
    "delta_amount_bid_algo4",
    "qimb1_avg",
    "qimb1_std",
    "qimb1_skew",
    "qimb1_kurt",
    "qimb10_avg",
    "qimb10_std",
    "qimb10_skew",
    "qimb10_kurt",
    "tick_return_avg",
    "tick_return_std",
    "tick_return_skew",
    "tick_return_kurt",
    "ask_amount10_chg_avg",
    "ask_amount10_chg_std",
    "ask_amount10_chg_skew",
    "ask_amount10_chg_kurt",
    "bid_amount10_chg_avg",
    "bid_amount10_chg_std",
    "bid_amount10_chg_skew",
    "bid_amount10_chg_kurt",
    "ask_amount10_ratio1_avg",
    "ask_amount10_ratio1_std",
    "ask_amount10_ratio1_skew",
    "ask_amount10_ratio1_kurt",
    "bid_amount10_ratio1_avg",
    "bid_amount10_ratio1_std",
    "bid_amount10_ratio1_skew",
    "bid_amount10_ratio1_kurt",
    "ask_amount10_ratio2_avg",
    "ask_amount10_ratio2_std",
    "ask_amount10_ratio2_skew",
    "ask_amount10_ratio2_kurt",
    "bid_amount10_ratio2_avg",
    "bid_amount10_ratio2_std",
    "bid_amount10_ratio2_skew",
    "bid_amount10_ratio2_kurt",
    "book10_ratio_avg",
    "book10_ratio_std",
    "book10_ratio_skew",
    "book10_ratio_kurt",
    "book10_ratio_chg_avg",
    "book10_ratio_chg_std",
    "book10_ratio_chg_skew",
    "book10_ratio_chg_kurt",
    "book10_rratio_avg",
    "book10_rratio_std",
    "book10_rratio_skew",
    "book10_rratio_kurt",
    "book10_rratio_chg_avg",
    "book10_rratio_chg_std",
    "book10_rratio_chg_skew",
    "book10_rratio_chg_kurt",
    "twap",
    "arrival_time_from_trans",
    "total_trades_from_trans",
    "twap_from_trans",
    "buy_amount_by_bsflag_from_trans",
    "sell_amount_by_bsflag_from_trans",
    "buy_amount_by_tick_from_trans",
    "sell_amount_by_tick_from_trans",
    "buy_amount_by_quote_from_trans",
    "sell_amount_by_quote_from_trans",
    "vwap",
    "qimb_1_close",
    "qimb_10",
    "timb_bsflag",
    "timb_tick",
    "timb_quote",
    "mpb",
    "tpb",
    "hml",
    "arrival_time_v3",
    "open_amount_from_trans",
    "close_amount_from_trans",
    "high_amount_from_trans",
    "low_amount_from_trans",
    "buy_price_sum_from_order",
    "buy_price_max_volume_from_order",
    "buy_price_max_from_order",
    "buy_price_min_from_order",
    "buy_price_second_moment_from_order",
    "buy_price_third_moment_from_order",
    "buy_price_fourth_moment_from_order",
    "buy_price_vw_second_moment_from_order",
    "buy_price_vw_third_moment_from_order",
    "buy_price_vw_fourth_moment_from_order",
    "buy_price_q50_from_order",
    "buy_order_count_from_order",
    "sell_price_sum_from_order",
    "sell_price_max_volume_from_order",
    "sell_price_max_from_order",
    "sell_price_min_from_order",
    "sell_price_second_moment_from_order",
    "sell_price_third_moment_from_order",
    "sell_price_fourth_moment_from_order",
    "sell_price_vw_second_moment_from_order",
    "sell_price_vw_third_moment_from_order",
    "sell_price_vw_fourth_moment_from_order",
    "sell_price_mean_from_order",
    "sell_price_q50_from_order",
    "sell_order_count_from_order",
    "all_price_q50_from_order",
    "all_price_q25_from_order",
    "all_price_q75_from_order",
    "buy_volume_sum_from_order",
    "buy_volume_second_moment_from_order",
    "buy_volume_third_moment_from_order",
    "buy_volume_fourth_moment_from_order",
    "sell_volume_sum_from_order",
    "sell_volume_second_moment_from_order",
    "sell_volume_third_moment_from_order",
    "sell_volume_fourth_moment_from_order",
    "buy_amount_sum_from_order",
    "buy_amount_second_moment_from_order",
    "buy_amount_third_moment_from_order",
    "buy_amount_fourth_moment_from_order",
    "sell_amount_sum_from_order",
    "sell_amount_second_moment_from_order",
    "sell_amount_third_moment_from_order",
    "sell_amount_fourth_moment_from_order",
    "buy_price_mean_from_order",
    "exlarge_buy_count2_from_order",
    "exlarge_buy_volume2_from_order",
    "exlarge_buy_amount2_from_order",
    "exlarge_sell_count2_from_order",
    "exlarge_sell_volume2_from_order",
    "exlarge_sell_amount2_from_order",
    "large_buy_count2_from_order",
    "large_buy_volume2_from_order",
    "large_buy_amount2_from_order",
    "large_sell_count2_from_order",
    "large_sell_volume2_from_order",
    "large_sell_amount2_from_order",
    "med_buy_count2_from_order",
    "med_buy_volume2_from_order",
    "med_buy_amount2_from_order",
    "med_sell_count2_from_order",
    "med_sell_volume2_from_order",
    "med_sell_amount2_from_order",
    "small_buy_count2_from_order",
    "small_buy_volume2_from_order",
    "small_buy_amount2_from_order",
    "small_sell_count2_from_order",
    "small_sell_volume2_from_order",
    "small_sell_amount2_from_order",
    "exlarge_buy_count_from_order",
    "exlarge_buy_volume_from_order",
    "exlarge_buy_amount_from_order",
    "exlarge_sell_count_from_order",
    "exlarge_sell_volume_from_order",
    "exlarge_sell_amount_from_order",
    "large_buy_count_from_order",
    "large_buy_volume_from_order",
    "large_buy_amount_from_order",
    "large_sell_count_from_order",
    "large_sell_volume_from_order",
    "large_sell_amount_from_order",
    "med_buy_count_from_order",
    "med_buy_volume_from_order",
    "med_buy_amount_from_order",
    "med_sell_count_from_order",
    "med_sell_volume_from_order",
    "med_sell_amount_from_order",
    "small_buy_count_from_order",
    "small_buy_volume_from_order",
    "small_buy_amount_from_order",
    "small_sell_count_from_order",
    "small_sell_volume_from_order",
    "small_sell_amount_from_order",
    "effe_buy_price_sum_from_order",
    "effe_buy_price_max_volume_from_order",
    "effe_buy_price_max_from_order",
    "effe_buy_price_min_from_order",
    "effe_buy_price_second_moment_from_order",
    "effe_buy_price_third_moment_from_order",
    "effe_buy_price_fourth_moment_from_order",
    "effe_buy_price_vw_second_moment_from_order",
    "effe_buy_price_vw_third_moment_from_order",
    "effe_buy_price_vw_fourth_moment_from_order",
    "effe_buy_price_mean_from_order",
    "effe_buy_price_q50_from_order",
    "effe_buy_order_count_from_order",
    "effe_sell_price_sum_from_order",
    "effe_sell_price_max_volume_from_order",
    "effe_sell_price_max_from_order",
    "effe_sell_price_min_from_order",
    "effe_sell_price_second_moment_from_order",
    "effe_sell_price_third_moment_from_order",
    "effe_sell_price_fourth_moment_from_order",
    "effe_sell_price_vw_second_moment_from_order",
    "effe_sell_price_vw_third_moment_from_order",
    "effe_sell_price_vw_fourth_moment_from_order",
    "effe_sell_price_mean_from_order",
    "effe_sell_price_q50_from_order",
    "effe_sell_order_count_from_order",
    "effe_all_price_q50_from_order",
    "effe_all_price_q25_from_order",
    "effe_all_price_q75_from_order",
    "effe_buy_volume_sum_from_order",
    "effe_buy_volume_second_moment_from_order",
    "effe_buy_volume_third_moment_from_order",
    "effe_buy_volume_fourth_moment_from_order",
    "effe_sell_volume_sum_from_order",
    "effe_sell_volume_second_moment_from_order",
    "effe_sell_volume_third_moment_from_order",
    "effe_sell_volume_fourth_moment_from_order",
    "effe_buy_amount_sum_from_order",
    "effe_buy_amount_second_moment_from_order",
    "effe_buy_amount_third_moment_from_order",
    "effe_buy_amount_fourth_moment_from_order",
    "effe_sell_amount_sum_from_order",
    "effe_sell_amount_second_moment_from_order",
    "effe_sell_amount_third_moment_from_order",
    "effe_sell_amount_fourth_moment_from_order",
    "effe_buy_volume_max_from_order",
    "effe_buy_volume_min_from_order",
    "effe_sell_volume_max_from_order",
    "effe_sell_volume_min_from_order",
    "quick_1s_cancel_buy_volume_sum_from_order",
    "quick_1s_cancel_buy_amount_sum_from_order",
    "quick_1s_cancel_sell_volume_sum_from_order",
    "quick_1s_cancel_sell_amount_sum_from_order",
    "quick_3s_cancel_buy_volume_sum_from_order",
    "quick_3s_cancel_buy_amount_sum_from_order",
    "quick_3s_cancel_sell_volume_sum_from_order",
    "quick_3s_cancel_sell_amount_sum_from_order",
    "big_3s_cancel_buy_volume_sum_from_order",
    "big_3s_cancel_buy_amount_sum_from_order",
    "big_3s_cancel_sell_volume_sum_from_order",
    "big_3s_cancel_sell_amount_sum_from_order",
    "cancel_buy_price_sum_from_order",
    "cancel_buy_price_max_volume_from_order",
    "cancel_buy_price_max_from_order",
    "cancel_buy_price_min_from_order",
    "cancel_buy_price_second_moment_from_order",
    "cancel_buy_price_third_moment_from_order",
    "cancel_buy_price_fourth_moment_from_order",
    "cancel_buy_price_vw_second_moment_from_order",
    "cancel_buy_price_vw_third_moment_from_order",
    "cancel_buy_price_vw_fourth_moment_from_order",
    "cancel_buy_price_mean_from_order",
    "cancel_buy_price_q50_from_order",
    "cancel_buy_order_count_from_order",
    "cancel_sell_price_sum_from_order",
    "cancel_sell_price_max_volume_from_order",
    "cancel_sell_price_max_from_order",
    "cancel_sell_price_min_from_order",
    "cancel_sell_price_second_moment_from_order",
    "cancel_sell_price_third_moment_from_order",
    "cancel_sell_price_fourth_moment_from_order",
    "cancel_sell_price_vw_second_moment_from_order",
    "cancel_sell_price_vw_third_moment_from_order",
    "cancel_sell_price_vw_fourth_moment_from_order",
    "cancel_sell_price_mean_from_order",
    "cancel_sell_price_q50_from_order",
    "cancel_sell_order_count_from_order",
    "cancel_all_price_q50_from_order",
    "cancel_all_price_q25_from_order",
    "cancel_all_price_q75_from_order",
    "cancel_buy_volume_sum_from_order",
    "cancel_buy_volume_second_moment_from_order",
    "cancel_buy_volume_third_moment_from_order",
    "cancel_buy_volume_fourth_moment_from_order",
    "cancel_sell_volume_sum_from_order",
    "cancel_sell_volume_second_moment_from_order",
    "cancel_sell_volume_third_moment_from_order",
    "cancel_sell_volume_fourth_moment_from_order",
    "cancel_buy_amount_sum_from_order",
    "cancel_buy_amount_second_moment_from_order",
    "cancel_buy_amount_third_moment_from_order",
    "cancel_buy_amount_fourth_moment_from_order",
    "cancel_sell_amount_sum_from_order",
    "cancel_sell_amount_second_moment_from_order",
    "cancel_sell_amount_third_moment_from_order",
    "cancel_sell_amount_fourth_moment_from_order",
    "cancel_buy_volume_max_from_order",
    "cancel_buy_volume_min_from_order",
    "cancel_sell_volume_max_from_order",
    "cancel_sell_volume_min_from_order",
    "buy_price_bia_mean_from_order",
    "sell_price_bia_mean_from_order",
    "buy_price_bia_square_mean_from_order",
    "sell_price_bia_square_mean_from_order",
    "buy_price_bia_vw_mean_from_order",
    "sell_price_bia_vw_mean_from_order",
    "buy_price_bia_prod_volume_sum_from_order",
    "sell_price_bia_prod_volume_sum_from_order",
    "buy_traded_volume_sum_from_order",
    "sell_traded_volume_sum_from_order",
    "buy_traded_amount_sum_from_order",
    "sell_traded_amount_sum_from_order",
    "buy_traded_act_buy_volume_sum_from_order",
    "buy_traded_act_sell_volume_sum_from_order",
    "sell_traded_act_buy_volume_sum_from_order",
    "sell_traded_act_sell_volume_sum_from_order",
    "buy_traded_times_sum_from_order",
    "sell_traded_times_sum_from_order",
    "buy_traded_price_mean_from_order",
    "sell_traded_price_mean_from_order",
    "buy_log_price_product_volume_sum_from_order",
    "sell_log_price_product_volume_sum_from_order",
    "buy_log_price_vw_second_moment_from_order",
    "sell_log_price_vw_second_moment_from_order",
    "log_price_product_volume_sum_from_order",
    "log_price_vw_second_moment_from_order",
    "buy_order_continious_time_mean_from_order",
    "sell_order_continious_time_mean_from_order",
    "buy_order_continious_time_max_from_order",
    "sell_order_continious_time_max_from_order",
    "buy_order_chasing_volume_mean_from_order",
    "sell_order_chasing_volume_mean_from_order",
    "buy_order_chasing_volume_max_from_order",
    "sell_order_chasing_volume_max_from_order",
    "high_ntrades_from_trans",
    "low_ntrades_from_trans",
    "act_buy_max_volume_from_trans",
    "act_sell_max_volume_from_trans",
    "act_buy_max_price_from_trans",
    "act_sell_max_price_from_trans",
    "act_buy_min_volume_from_trans",
    "act_sell_min_volume_from_trans",
    "act_buy_min_price_from_trans",
    "act_sell_min_price_from_trans",
    "buy_price_sum_from_trans",
    "buy_price_max_volume_from_trans",
    "buy_price_max_from_trans",
    "buy_price_min_from_trans",
    "buy_price_second_moment_from_trans",
    "buy_price_third_moment_from_trans",
    "buy_price_fourth_moment_from_trans",
    "buy_price_vw_second_moment_from_trans",
    "buy_price_vw_third_moment_from_trans",
    "buy_price_vw_fourth_moment_from_trans",
    "buy_price_mean_from_trans",
    "buy_price_q50_from_trans",
    "buy_order_count_from_trans",
    "sell_price_sum_from_trans",
    "sell_price_max_volume_from_trans",
    "sell_price_max_from_trans",
    "sell_price_min_from_trans",
    "sell_price_second_moment_from_trans",
    "sell_price_third_moment_from_trans",
    "sell_price_fourth_moment_from_trans",
    "sell_price_vw_second_moment_from_trans",
    "sell_price_vw_third_moment_from_trans",
    "sell_price_vw_fourth_moment_from_trans",
    "sell_price_mean_from_trans",
    "sell_price_q50_from_trans",
    "sell_order_count_from_trans",
    "all_price_q50_from_trans",
    "all_price_q25_from_trans",
    "all_price_q75_from_trans",
    "buy_volume_sum_from_trans",
    "buy_volume_second_moment_from_trans",
    "buy_volume_third_moment_from_trans",
    "buy_volume_fourth_moment_from_trans",
    "sell_volume_sum_from_trans",
    "sell_volume_second_moment_from_trans",
    "sell_volume_third_moment_from_trans",
    "sell_volume_fourth_moment_from_trans",
    "buy_amount_sum_from_trans",
    "buy_amount_second_moment_from_trans",
    "buy_amount_third_moment_from_trans",
    "buy_amount_fourth_moment_from_trans",
    "sell_amount_sum_from_trans",
    "sell_amount_second_moment_from_trans",
    "sell_amount_third_moment_from_trans",
    "sell_amount_fourth_moment_from_trans",
    "buy_volume_max_from_trans",
    "buy_volume_min_from_trans",
    "sell_volume_max_from_trans",
    "sell_volume_min_from_trans",
]

excluded_bars = [
    "data_source",
    "arrival_time",
    "arrival_time_from_trans",
    "arrival_time_v3",
    "mpb",
    "tpb",
    "hml",
    "iopv",
    "twap",
    "twap_from_trans",
    "total_trades_from_trans",
]


def get_bars(feature_set: str = 'v2_agg') -> list[str]:
    """get list of bars(features) from name.

    Args:
        feature_set (str, optional): name of features set. Defaults to 'v2_agg'.

    Raises:
        ValueError: _description_

    Returns:
        list[str]: list of bars in the feature set.
    """    
    if feature_set == 'v2_agg':
        bars = sorted(set(bar_v2) - set(excluded_bars))
        agg = ['mean', 'std']
        import itertools
        return [f"{bar}_{a}" for bar, a in itertools.product(bars, agg)]
    elif feature_set == 'v2':
        bars = sorted(set(bar_v2) - set(excluded_bars))
        return bars

    else:
        raise ValueError(f"feature_set {feature_set} not supported")
    

def split_date_ranges(
    calendar: list[str], 
    milestone: str = '2020-01-01', 
    n_train: int = 1000, 
    n_eval: int = 30, 
    n_lag: int = 5, 
    n_test: int = 0
    ) -> tuple[tuple[str, str], tuple[str, str], Optional[tuple[str, str]]]:
    """Extract split data ranges from calendar on milestone.

    Args:
        calendar (list[str]): list of all trading dates in format of YYYY-MM-DD.
        milestone (str, optional): dates behind milestone are out of sample dates. Defaults to '2020-01-01'.
        n_train (int, optional): number of training dates. Defaults to 1000.
        n_eval (int, optional): number of eval dates. Defaults to 30.
        n_lag (int, optional): number of lag dates (should be skipped) between eval and test dates. Defaults to 5.
        n_test (int, optional): number of test dates. Defaults to 0.

    Raises:
        ValueError: _description_

    Returns:
        tuple[tuple[str, str], tuple[str, str], Optional[tuple[str, str]]]: _description_
    """    
    import bisect
    i = bisect.bisect(calendar, milestone)
    _l = i - n_lag - n_eval
    if _l < 0:
        raise ValueError(f"not enough data for eval, {n_eval} is too big.")
    eval_dates = calendar[_l: i-n_lag]
    _l = max(i-n_eval - n_train - n_lag, 0)
    train_dates = calendar[_l: i-n_eval-n_lag]
    eval_range = (eval_dates[0], eval_dates[-1])
    train_range = (train_dates[0], train_dates[-1])
    if n_test != 0:
        test_dates = calendar[i:i+n_test]
        test_range = (test_dates[0], test_dates[-1])
    else:
        test_range = None
    return train_range, eval_range, test_range


class ProdsAvailable(str, Enum):
    PROD_0930 = '0930'
    PROD_0930_1H = '0930_1h'
    PROD_1030 = '1030'
    PROD_1030_1H = '1030_1h'
    PROD_1300 = '1300'
    PROD_1300_1H = '1300_1h'
    PROD_1400 = '1400'
    PROD_1400_1H = '1400_1h'


def list_prods() -> list[str]:
    return [prod.value for prod in ProdsAvailable]

def get_prod_data_config(
    prod: ProdsAvailable = ProdsAvailable.PROD_1030
    ) -> DictConfig:
    if prod == ProdsAvailable.PROD_0930:
        cfg = {
            'slot': "0930",
            'x_begin': "1300", 
            'x_end': "1500",
            'freq': 10,
            'ret_prefix': "next_rtn_v2v_",
            'ret_durations': ['1D', '2D', '5D'],
            'delay': 5,
            }
    elif prod == ProdsAvailable.PROD_0930_1H:
        cfg = {
            'slot': "0930",
            'x_begin': "1300", 
            'x_end': "1500",
            'freq': 10,
            'ret_prefix': "next_rtn_v2v_",
            'ret_durations': ['1h'],
            'delay': 5,
            }
    elif prod == ProdsAvailable.PROD_1030:
        cfg = {
            'slot': "1030",
            'x_begin': "0930", 
            'x_end': "1030",
            'freq': 10,
            'ret_prefix': "rtn_v2v_",
            'ret_durations': ['1D', '2D', '5D'],
            'delay': 5,
            }
    elif prod == ProdsAvailable.PROD_1030_1H:
        cfg = {
            'slot': "1030",
            'x_begin': "0930", 
            'x_end': "1030",
            'freq': 10,
            'ret_prefix': "rtn_v2v_",
            'ret_durations': ['1h'],
            'delay': 5,
            }
    elif prod == ProdsAvailable.PROD_1300:
        cfg = {
            'slot': "1300",
            'x_begin': "0930", 
            'x_end': "1300",
            'freq': 10,
            'ret_prefix': "rtn_v2v_",
            'ret_durations': ['1D', '2D', '5D'],
            'delay': 5,
            }
    elif prod == ProdsAvailable.PROD_1300_1H:
        cfg = {
            'slot': "1300",
            'x_begin': "0930", 
            'x_end': "1300",
            'freq': 10,
            'ret_prefix': "rtn_v2v_",
            'ret_durations': ['1h'],
            'delay': 5,
            }
    elif prod == ProdsAvailable.PROD_1400:
        cfg = {
            'slot': "1400",
            'x_begin': "1030", 
            'x_end': "1400",
            'freq': 10,
            'ret_prefix': "rtn_v2v_",
            'ret_durations': ['1D', '2D', '5D'],
            'delay': 5,
            }
    elif prod == ProdsAvailable.PROD_1400_1H:
        cfg = {
            'slot': "1400",
            'x_begin': "1030", 
            'x_end': "1400",
            'freq': 10,
            'ret_prefix': "rtn_v2v_",
            'ret_durations': ['1h'],
            'delay': 5,
            } 
    else:
        raise ValueError(f"prod {prod} not supported")

    _prefix = f"{cfg['ret_prefix']}{int(cfg['slot'])+int(cfg['delay']):04d}"
    cfg['y_cols'] = [f"{_prefix}_{_d}" for _d in cfg['ret_durations']]
    # cfg['experiment_dir'] = f"prod_runs/{cfg['slot']}"
    return OmegaConf.create(cfg)
        

_default_config = {
    'feature_set': 'v2_agg',
    'd_in': 278,
    'model': 'GPT_small',
    'epochs': 20,
    'patience': 6,
    'universe': 'euniv_largemid',
    'n_train': 1000,
    'n_eval': 30,
    'n_lag': 5,
    'n_test': 0,
    'n_latest': 3,
}

def get_training_config(prod: Optional[ProdsAvailable] = None, milestone: Optional[str] = None) -> TrainArguments:
    if prod is None:
        raise ValueError("prod must be specified")
    if milestone is None:
        raise ValueError("milestone must be specified")
    
    # Get absolute path to the directory this script (or module) is in
    # script_dir = os.path.dirname(os.path.realpath(__file__))
    prod_available = list_prods()
    if prod not in prod_available:
        raise ValueError(f"prod {prod} not available, "
                         f"available prods: {prod_available}")
    import pickle
    from pathlib import Path

    # prod_cfg = OmegaConf.load(os.path.join(script_dir, f'pred/{prod}.yaml'))
    prod_cfg = get_prod_data_config(prod=prod)
    # meta_cfg = OmegaConf.load(os.path.join(script_dir, 'config.yaml'))
    meta_cfg = _default_config
    cfg = OmegaConf.merge(prod_cfg, meta_cfg)
    
    # cld = pickle.load(open(os.path.join(script_dir, 'calendar.pkl'), 'rb'))
    cld_path = os.getenv("CALENDAR_PATH") or ""
    
    if os.path.isfile(cld_path) is False:
        raise ValueError("CALENDAR_PATH must be specified")
    else:
        cld = pickle.load(open(cld_path, 'rb'))

    train_range, eval_range, test_range = split_date_ranges(
        cld, milestone=milestone, 
        n_train=cfg.n_train, 
        n_eval=cfg.n_eval, 
        n_lag=cfg.n_lag, 
        n_test=cfg.n_test
        )
    data_dir = os.getenv('DATASET_DIR')
    if  data_dir is None:
        raise ValueError("DATASET_DIR must be specified")
    else:
        data_dir = Path(data_dir)
        save_dir = Path(os.getenv('SAVE_DIR') or "runs")
        args = TrainArguments(
            prod=prod,
            save_dir=save_dir,
            dataset_dir=data_dir,
            milestone=milestone,
            universe=cfg.universe,
            x_columns=get_bars(cfg.feature_set),
            x_begin=cfg.x_begin,
            x_end=cfg.x_end,
            freq_in_min=cfg.freq,
            y_columns=cfg.y_cols,
            y_slots=cfg.slot,
            model=cfg.model,
            device='cuda',
            train_date_range=train_range,
            eval_date_range=eval_range,
            test_date_range=test_range,
            epochs=cfg.epochs,
            lr=5.0e-5,
            weight_decay=1.0e-3,
            patience=cfg.patience,
            normalizer='zscore',
            seed=42,
            monitor_metric='loss',
            monitor_mode='min',
            train_batch_size=1024,
            eval_batch_size=2048,
            test_batch_size=2048,
            dataloader_drop_last=False,
        )
        return args


def get_inference_config(prod: Optional[ProdsAvailable] = None) -> InferenceArguments:
    if prod is None:
        raise ValueError("prod must be specified")
    
    # Get absolute path to the directory this script (or module) is in
    # script_dir = os.path.dirname(os.path.realpath(__file__))
    prod_available = list_prods()
    if prod not in prod_available:
        raise ValueError(f"prod {prod} not available, "
                         f"available prods: {prod_available}")

    prod_cfg = get_prod_data_config(prod=prod)
    # meta_cfg = OmegaConf.load(os.path.join(script_dir, 'config.yaml'))
    meta_cfg = _default_config
    cfg = OmegaConf.merge(prod_cfg, meta_cfg)
    
    from pathlib import Path
    data_dir = os.getenv('DATASET_DIR')
    if  data_dir is None:
        raise ValueError("DATASET_DIR must be specified")
    else:
        data_dir = Path(data_dir)
        save_dir = Path(os.getenv('SAVE_DIR') or "runs")
        args = InferenceArguments(
            prod=prod,
            save_dir=save_dir,
            dataset_dir=data_dir,
            universe=cfg.universe,
            x_columns=get_bars(cfg.feature_set),
            x_begin=cfg.x_begin,
            x_end=cfg.x_end,
            freq_in_min=cfg.freq,
            y_columns=cfg.y_cols,
            y_slots=cfg.slot,
            model=cfg.model,
            n_latest=3,
            device='cuda',
        )
        return args