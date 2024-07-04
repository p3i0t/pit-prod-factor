# pit

Project of ``pit`` factors. This project maintains the whole lifecycle to generate a production alpha factor, including: download raw data, data processing, training pipeline, inference pipeline. A cli is exposed, zero extra code is required to use it.

All data, model checkpoints etc are stored in ``pit`` home directory specified by environment variable `PIT_DIR`, which defaults to `~/.pit`.

## Usage via CLI

This package exposes a cli, run:

``
pit --help
``

to get the full command list.


### Update trading calendar

``pit`` maintains a local trading calendar in ``PIT_DIR``, run:

```shell
pit update-calendar 
```

to update the calendar, the calendar starts from 2015-01-01, and ends on the last day of this year.

  **Hint: Since the trading calendar is updated in an unpredictable manner due to adjustment of public holidays in China. It is advised to update the calendar every day.**

### Download data

download all data items (return/lag_return/univ/bar_1m) from 2017-01-01 to today:

```shell
pit download -t all --begin 2017-01-01 --end today -v
```

### Dataset Generation

```shell
pit generate-dataset
```

The dataset will be stored in ``PIT_DIR/dataset``.

### Training

train single model given the production name and milestone date, e.g. prod 1030 and milestone 2024-02-01:

```shell
pit train-single -p 1030 -m 2024-02-01
```

``milestone``  is the date that the model is supposed to be available. Training one single model on a given ``milestone`` only involves information earlier than the ``milestone``.


### Online factor generation

run ``pit infer-online`` to generate pit factors, e.g. the following command generate pit factor at 1030 on date today (other acceptable date argument can be in format like 20240401, 2024-04-01) in verbose mode.

```shell
pit infer-online --prod 1030 --date today -v
```

Make sure that the ``production`` data is ready when running the command, and the online ``production`` data is fetched from ``datareader``.

