# pit

Project of ``pit`` factors. This project maintains the whole lifecycle to generate a production alpha factor, including: download raw data, data processing, training pipeline, inference pipeline. A cli is exposed, zero extra code is required to use it.

All data items (raw or processed), model checkpoints etc are stored in ``pit`` home directory specified by environment variable `PIT_DIR`, which defaults to `~/.pit`.

## Usage via CLI

This package exposes a cli, run:

``
pit --help
``

to get the full command list.

### Initialization

run 

```pit init``` 

to initialize the directory structure in ``PIT_DIR``.

You have to run this again if a new ``PIT_DIR`` is set.

### Update trading calendar

``pit`` maintains a local trading calendar in ``PIT_DIR``, run:

```shell
pit update-calendar 
```

to update the calendar, the calendar starts from 2015-01-01, and ends on the last day of this year.

  **Hint: Since the trading calendar is updated in an unpredictable manner due to adjustment of public holidays in China. It is advised to update the calendar every day.**

### Online factor generation

run ``pit infer-online`` to generate pit factors, e.g. the following command generate pit factor at 1030 on date today (other acceptable date argument can be in format like 20240401, 2024-04-01) in verbose mode.

```shell
pit infer-online --prod 1030 --date today -v
```

Make sure that the production data is ready when running the command, and the online production data is fetched from ``datareader``.

### Training

train single model given the production name and milestone date, e.g. prod 1030 and milestone 2024-02-01:

```shell
pit train-single -p 1030 -m 2024-02-01
```

``milestone``  is the date that the model is supposed to be available. Training one single model on a given ``milestone`` only involves information earlier than the ``milestone``.

### Dataset Generation

Assuming 1-minute bars, universe, return data items are all downloaded in ``PIT_DIR``, first downsample the 1-minute bars:

```shell
pit download10 --n_jobs 10 --cpu_per_task 3 -v
```

Then merge the processed bars with return:

```shell
pit merge10-v2 --n_jobs 10 --cpu_per_task 3
```

The dataset will be stored in ``PIT_DIR/dataset``.
