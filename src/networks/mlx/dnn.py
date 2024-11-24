import mlx.core as mx
import mlx.nn as nn
import numpy as np
from .network import Network

class DNN(Network):
  @staticmethod
  def get_network_head(inp, output_dim):
    return [
      nn.BatchNorm(inp[0]),
      nn.Linear(inp[0], 256),
      nn.BatchNorm(256),
      nn.Dropout(p=0.1),
      nn.Linear(256, 128),
      nn.BatchNorm(128),
      nn.Dropout(p=0.1),
      nn.Linear(128, 64),
      nn.BatchNorm(64),
      nn.Dropout(p=0.1),
      nn.Linear(64, 32),
      nn.BatchNorm(32),
      nn.Dropout(p=0.1),
      nn.Linear(32, output_dim),
    ]

  def train_on_batch(self, x, y):
    x = np.array(x).reshape((-1, self.input_dim))
    return super().train_on_batch(x, y)

  def predict(self, sample):
    sample = np.array(sample).reshape((1, self.input_dim))
    return super().predict(sample)