from .network import Network
import torch
import numpy as np

class CNN_LSTMNetwork(Network):
    def __init__(self, *args, num_steps=1, **kwargs):
        self.num_steps = num_steps
        super().__init__(*args, **kwargs)

    @staticmethod
    def get_network_head(inp, output_dim):
        kernel_size = 2
        return torch.nn.Sequential(
            torch.nn.BatchNorm1d(inp[0]),
            torch.nn.Conv1d(inp[0], 32, kernel_size),
            torch.nn.Dropout(p=0.1),
            torch.nn.Conv1d(32, 64, kernel_size),
            torch.nn.Dropout(p=0.1),
            LSTMModule(inp[1] - 2, 128, batch_first=True, use_last_only=True),
            torch.nn.BatchNorm1d(128),
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(128, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(64, 32),
            torch.nn.BatchNorm1d(32),
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(32, output_dim),
        )

    def train_on_batch(self, x, y):
        x = np.array(x).reshape((-1, self.num_steps, self.input_dim))
        return super().train_on_batch(x, y)

    def predict(self, sample):
        sample = np.array(sample).reshape((1, self.num_steps, self.input_dim))
        return super().predict(sample)

class LSTMModule(torch.nn.LSTM):
  def __init__(self, *args, use_last_only=False, **kwargs):
    super().__init__(*args, **kwargs)
    self.use_last_only = use_last_only

  def forward(self, x):
    output, (h_n, _) = super().forward(x)
    if self.use_last_only:
      return h_n[-1]
    return output