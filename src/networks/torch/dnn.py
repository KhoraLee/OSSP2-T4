from .network import Network
import torch
import numpy as np

class DNN(Network):
  @staticmethod
  def get_network_head(inp, output_dim):
    return torch.nn.Sequential(
      torch.nn.BatchNorm1d(inp[0]),
      torch.nn.Linear(inp[0], 256),
      torch.nn.BatchNorm1d(256),
      torch.nn.Dropout(p=0.1),
      torch.nn.Linear(256, 128),
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
    x = np.array(x).reshape((-1, self.input_dim))
    return super().train_on_batch(x, y)

  def predict(self, sample):
    sample = np.array(sample).reshape((1, self.input_dim))
    return super().predict(sample)