from typing import Union, Sequence
from dataclasses import dataclass
import numpy as np
import torch
import pandas as pd
from dlkit import SuperModule
from dlkit.models import GPT

from utils import compute_ic


@dataclass
class TrainMeta:
    """Meta data of train set, used for normalization (preprocessing).
    """
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray


class XnY1Model(SuperModule):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        agg: str = 'last',
        train_meta: TrainMeta = None,
        lr=3e-5,
        weight_decay=1e-3,
        x_labels: np.ndarray = None, # 2d array: (S, D), S for sequence time slots, D for input features.
        y_labels: np.ndarray = None, # 1d array: (Y), Y: number of output.
        model_available_date: str = None,
        device='cuda',
        verbose=True,
        **params,
    ) -> None:
        super().__init__()
        device = 'cuda' if device == 'cuda' and torch.cuda.is_available() else 'cpu'
        self.model = GPT(
            d_in=d_in,
            d_out=d_out,
            d_model=256,
            dim_feedforward=512,
            n_head=4,
            dropout=0.3,
            n_layers=2,
            agg=agg,
        ).to(device=device)
        self.train_meta = train_meta
        self.lr = lr
        self.weight_decay = weight_decay
        self.y_labels = y_labels

        self.x_labels: np.ndarray = x_labels
        self.y_labels: np.ndarray = y_labels
        self.device = device
        self.verbose = verbose

        self._n_parameters = None
        self.available_date = model_available_date

        if self.train_meta is not None:
            self.train_meta.x_mean = torch.Tensor(self.train_meta.x_mean.copy()).to(self.device)
            self.train_meta.x_std = torch.Tensor(self.train_meta.x_std.copy()).to(self.device)
            self.train_meta.y_mean = torch.Tensor(self.train_meta.y_mean.copy()).to(self.device)
            self.train_meta.y_std = torch.Tensor(self.train_meta.y_std.copy()).to(self.device)

        # note that all the following dates should be earlier than `self.available_date`
        self.train_dates: Sequence[str] = None
        self.validation_dates: Sequence[str] = None
        self.lag_dates: Sequence[str] = None
        for k, v in params.items():
            setattr(self, k, v)
        self.optimizer = self.configure_optimizers()

    @property
    def n_parameters(self):
        if self._n_parameters is None:
            self._n_parameters = sum(param.numel() for param in self.model.parameters())
        return self._n_parameters

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

    def state_dict(self) -> dict:
        return {
            'available_date': self.available_date,
            'model': self.model.state_dict(),
            'x_mean': self.train_meta.x_mean,
            'x_std': self.train_meta.x_std,
            'y_mean': self.train_meta.y_mean,
            'y_std': self.train_meta.y_std,
            'x_labels': self.x_labels,
            'y_labels': self.y_labels,
            'train_dates': self.train_dates,
            'validation_dates': self.validation_dates,
            'lag_dates': self.lag_dates,
        }

    def load_state_dict(self, state_dict: dict, strict: bool = True):
        self.available_date = state_dict['available_date']
        train_meta = TrainMeta(
            x_mean = state_dict['x_mean'].to(self.device),
            x_std = state_dict['x_std'].to(self.device),
            y_mean = state_dict['y_mean'].to(self.device),
            y_std = state_dict['y_std'].to(self.device),
        )
        self.train_meta = train_meta
        self.train_dates = state_dict['train_dates']
        self.validation_dates = state_dict['validation_dates']
        self.lag_dates = state_dict['lag_dates']
        self.model.load_state_dict(state_dict=state_dict['model'], strict=strict)

    def _normalize_fillna(self, x, x_mean, x_std) -> torch.Tensor:
        # normalize and fillna
        x = (x - x_mean) / x_std
        x = torch.nan_to_num(x, nan=0.0)
        return x

    def on_batch_start(self, batch, batch_idx):
        batch['x'] = torch.Tensor(batch['x']).to(self.device, non_blocking=True)
        batch['y'] = torch.Tensor(batch['y']).to(self.device, non_blocking=True)
        batch['x'] = self._normalize_fillna(batch['x'], self.train_meta.x_mean, self.train_meta.x_std)
        batch['y'] = self._normalize_fillna(batch['y'], self.train_meta.y_mean, self.train_meta.y_std)

    def on_training_batch_start(self, batch, batch_idx):
        return self.on_batch_start(batch=batch, batch_idx=batch_idx)

    def on_validation_batch_start(self, batch, batch_idx):
        return self.on_batch_start(batch=batch, batch_idx=batch_idx)

    def on_test_batch_start(self, batch, batch_idx):
        return self.on_batch_start(batch=batch, batch_idx=batch_idx)

    def training_step(self, batch, batch_idx) -> dict:
        pred: torch.Tensor = self.model(batch['x'])
        loss = (pred - batch['y']).pow(2).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {'train_mse': float(loss)}

    def validation_step(self, batch, batch_idx) -> dict:
        pred: torch.Tensor = self.model(batch['x'])
        loss = (pred - batch['y']).pow(2).mean()
        y_cols = list(self.y_labels)
        pred_cols = [f"pred_{c}" for c in y_cols]

        df = pd.DataFrame()
        df['date'] = batch['date']
        df['symbol'] = batch['symbol']
        df[pred_cols] = pred.detach().cpu().numpy()
        df[y_cols] = batch['y'].detach().cpu().numpy()
        return {
            'validation_mse': float(loss),
            'validation_df': df
        }

    def _process_epoch_log(self, epoch_log: dict, prefix: str) -> dict:
        """merge batch logs and compute IC."""
        _log = {}
        # collapse list of step logs of one epoch
        for k, v in epoch_log.items():
            if k.endswith('mse'):
                _log[k] = np.mean(v)
            elif k.endswith('df'):
                df = pd.concat(v)
                y_cols = list(self.y_labels)
                pred_cols = [f"pred_{c}" for c in y_cols]
                for a, b in zip(pred_cols, y_cols):
                    _log[f"{prefix}_{b}_IC"] = compute_ic(df, [a], b).mean().iloc[0]
            else:
                raise TypeError(f"Type of element {v[0]} is not supported.")
        return _log

    def training_epoch_end(self, epoch_log: dict):
        return self._process_epoch_log(epoch_log=epoch_log, prefix='train')

    def validation_epoch_end(self, epoch_log: dict):
        return self._process_epoch_log(epoch_log=epoch_log, prefix='validation')

    def predict_step(self, x: Union[torch.Tensor, np.ndarray], out_format='numpy'):
        self.model.eval()
        if isinstance(x, np.ndarray):
            x = torch.Tensor(x).to(device=self.device)
        x = self._normalize_fillna(x, self.train_meta.x_mean, self.train_meta.x_std)
        with torch.no_grad():
            pred = self.model(x)
        o = pred.cpu()
        if out_format == 'numpy':
            return o.numpy()
        elif out_format == 'torch':
            return o
