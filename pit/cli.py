import os
import click
# from typing import Literal


# @click.command()
# @click.option('--source', default='bopu', help='source of calendar data')
# def update_calendar(source: Literal['akshare', 'bopu'] = 'bopu'):
#     import os
#     import pickle
#     script_directory = os.path.dirname(os.path.abspath(__file__))
#     if source == "akshare":
#         save_path = os.path.join(script_directory, 'ak_calendar.pkl')
#         import akshare as ak
#         df = ak.tool_trade_date_hist_sina()
#         trade_dates = [d.strftime('%Y-%m-%d') for d in sorted(df['trade_date']) if d.strftime('%Y-%m-%d') > '2017-01-01']
#         pickle.dump(trade_dates, open(save_path, 'wb'))
#     else:
#         try:
#             import genutils as gu  # noqa
#         except ImportError:
#             gu = None
#         trade_dates = gu.get_trade_dates()
#         save_path = os.path.join(script_directory, 'bopu_calendar.pkl')
#     return trade_dates

@click.command()
@click.option('--prod', default='1030', help='product to train')
@click.option('--milestone', default='today', help='milestone date of the model to train.')
def train_single(prod, milestone):
    """cli command to train single model of given prod and milestone.
    """
    from pit import get_training_config, TrainPipeline, list_prods
    os.environ['DATASET_DIR'] = '/data2/private/wangxin/dataset/10m_v2'
    os.environ['CALENDAR_PATH'] = 'calendar.pkl'
    os.environ['SAVE_DIR'] = 'pit_runs'
    
    valid_prods = list_prods()
    if prod not in valid_prods:
        raise ValueError(f"prod {prod} not in {valid_prods}")
    args = get_training_config(prod=prod, milestone=milestone)
    pipe = TrainPipeline(args)
    pipe.run()
    
    
# @click.command()
# @click.option('--prod', default='1030', help='product to infer')
# @click.option('--infer_date', default='today', help='inference date ')
# @click.option('--mode', default='offline', help='inference mode')
# def inference(prod, infer_date):
#     """cli command to train single model of given prod and milestone.
#     """
#     from pit import get_inference_config, list_prods
#     from pit.inference import infer, InferenceDataSource
    
#     os.environ['DATASET_DIR'] = '/data2/private/wangxin/dataset/10m_v2'
#     os.environ['CALENDAR_PATH'] = 'calendar.pkl'
#     os.environ['SAVE_DIR'] = 'pit_runs'
    
#     valid_prods = list_prods()
#     if prod not in valid_prods:
#         raise ValueError(f"prod {prod} not in {valid_prods}")
#     args = get_inference_config(prod=prod)
#     infer(args=args, infer_date=infer_date, mode=InferenceDataSource.offline)

@click.group()
def pit():
    """Manage my hahaha package."""
    click.echo("Alpha Signals Generator of Pit.")
    # Create the ~/.pit directory if it doesn't exist
    pit_dir = os.path.join(os.path.expanduser("~"), ".pit")
    if not os.path.exists(pit_dir):
        os.makedirs(pit_dir)
        click.echo(f"Created directory: {pit_dir}")
    else:
        pass
        # click.echo(f"Directory already exists: {pit_dir}")
    


# pit.add_command(data)
pit.add_command(train_single)

if __name__ == "__main__":
    pit()