import mlx.core as mx
import mlx.nn as nn
import numpy as np
from .network import Network

class CNN(Network):
    def __init__(self, *args, num_steps=1, **kwargs):
        self.num_steps = num_steps
        super().__init__(*args, **kwargs)

    @staticmethod
    def get_network_head(inp, output_dim):
      kernel_size = 2
      return [
        nn.BatchNorm(inp[1]),
        nn.Conv1d(inp[1], 1, kernel_size),
        nn.BatchNorm(1),
        Flatten(),
        nn.Dropout(p=0.1),
        nn.Linear(inp[0] - (kernel_size - 1), 128),
        nn.BatchNorm(128),
        nn.Dropout(p=0.1),
        nn.Linear(128, 64),
        nn.BatchNorm(64),
        nn.Dropout(p=0.1),
        nn.Linear(64, 32),
        nn.BatchNorm(32),
        nn.Dropout(p=0.1),
        nn.Linear(32, output_dim)
    ]

    def train_on_batch(self, x, y):
        x = np.array(x).reshape((-1, self.num_steps, self.input_dim))
        return super().train_on_batch(x, y)

    def predict(self, sample):
        sample = np.array(sample).reshape((1, self.num_steps, self.input_dim))
        return super().predict(sample)

class Flatten(nn.Module):
  def __call__(self, x):
    return mx.reshape(x, (x.shape[0], -1))