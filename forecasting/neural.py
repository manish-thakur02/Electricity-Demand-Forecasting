"""Use dense LSTM kernels for uniform windows, retaining variable-length fallback."""
import torch
from torch import nn
from pytorch_forecasting.models.nn.rnn import LSTM
from pytorch_forecasting.models.rnn import RecurrentNetwork
from pytorch_forecasting import TemporalFusionTransformer


class UniformLengthLSTM(LSTM):
    """Equivalent fixed-length recurrence without expensive packing on CPU.

    No padded step is consumed: truncate to the common positive length. Unequal,
    empty or already-packed inputs retain the upstream implementation.
    """
    def forward(self, x, hx=None, lengths=None, enforce_sorted=True):
        if isinstance(x, torch.Tensor) and lengths is not None:
            length = int(lengths[0])
            if length > 0 and bool(torch.all(lengths == length)):
                time_axis = 1 if self.batch_first else 0
                if length <= x.size(time_axis):
                    values = x[:, :length] if self.batch_first else x[:length]
                    hidden = None if hx is None else tuple(v.contiguous() for v in hx)
                    return nn.LSTM.forward(self, values.contiguous(), hx=hidden)
        return super().forward(x, hx=hx, lengths=lengths, enforce_sorted=enforce_sorted)


def uniform_lstm(original):
    # Preserve both initial weights and the RNG stream used by subsequent layers
    # and training. Replacement does not change architecture or parameter names.
    with torch.random.fork_rng(devices=[]):
        replacement = UniformLengthLSTM(original.input_size, original.hidden_size,
            num_layers=original.num_layers, bias=original.bias,
            batch_first=original.batch_first, dropout=original.dropout,
            bidirectional=original.bidirectional, proj_size=original.proj_size)
    replacement.load_state_dict(original.state_dict())
    replacement.train(original.training)
    return replacement


class FixedWindowRecurrentNetwork(RecurrentNetwork):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rnn = uniform_lstm(self.rnn)


class FixedWindowTemporalFusionTransformer(TemporalFusionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lstm_encoder = uniform_lstm(self.lstm_encoder)
        self.lstm_decoder = uniform_lstm(self.lstm_decoder)
