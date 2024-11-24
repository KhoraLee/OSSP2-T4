import abc
import os

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from ..base_network import BaseNetwork

class Network(BaseNetwork):
  def __init__(self, input_dim=0, output_dim=0, lr=0.001, shared_network=None, activation='relu', loss='mse'):
    super().__init__(input_dim, output_dim, lr, shared_network, activation, loss)
    self.layers = self.head
    if self.activation == 'linear':
      pass
    elif self.activation == 'relu':
      self.layers.append(nn.ReLU())
    elif self.activation == 'leaky_relu':
      self.layers.append(nn.LeakyReLU())
    elif self.activation == 'sigmoid':
      self.layers.append(nn.Sigmoid())
    elif self.activation == 'tanh':
      self.layers.append(nn.Tanh())
    elif self.activation == 'softmax':
      self.layers.append(nn.Softmax(axis=1))
    self.model = MLXModel(self.layers)
    self.model.apply(nn.init.normal())

    self.optimizer = optim.RMSprop(learning_rate=self.lr)
    self.optimizer = optim.Adam(learning_rate=self.lr)
    self.criterion = None
    if loss == 'mse':
      self.criterion = nn.losses.mse_loss
    elif loss == 'binary_crossentropy':
      self.criterion = nn.losses.binary_cross_entropy

  def predict(self, sample):
    with self.lock:
      self.model.eval()
      x = mx.array(sample, dtype=mx.float32)
      pred = self.model(x)
      return np.array(pred).flatten()

  # @mx.compile
  def train_on_batch(self, x, y):
    def loss_fn(x, y):
        y_pred = self.model(x)
        return self.criterion(y_pred, y)

    loss = 0.
    with self.lock:
      self.model.train()
      x = mx.array(x, dtype=mx.float32)
      y = mx.array(y, dtype=mx.float32)

      _loss, grads = nn.value_and_grad(self.model, loss_fn)(x, y)
      self.optimizer.update(self.model, grads)
      loss += _loss.item()
    return loss

  @classmethod
  def get_shared_network(cls, net='dnn', num_steps=1, input_dim=0, output_dim=0):
    from networks import CNN, DNN, LSTMNetwork
    if net == 'dnn':
      return DNN.get_network_head((input_dim,), output_dim)
    elif net == 'lstm':
      return LSTMNetwork.get_network_head((num_steps, input_dim), output_dim)
    elif net == 'cnn':
      return CNN.get_network_head((num_steps, input_dim), output_dim)

  @staticmethod
  @abc.abstractmethod
  def get_network_head(inp, output_dim):
    pass

  def save_model(self, model_path):
    if model_path is not None and self.model is not None:
      params_dict = {}
      for i, layer in enumerate(self.layers):
        layer_params = layer.parameters()
        for name, param in layer_params.items():
          params_dict[f"layer_{i}_{name}"] = param

      mx.save_safetensors(model_path, params_dict)
  def load_model(self, model_path):
    if model_path is not None and os.path.exists(model_path + ".safetensors"):
        loaded_params = mx.load(model_path + ".safetensors")
        # Reconstruct the layer parameters
        for i, layer in enumerate(self.layers):
            layer_params = {}
            for name, param in loaded_params.items():
                if name.startswith(f"layer_{i}_"):
                    param_name = name.replace(f"layer_{i}_", "")
                    layer_params[param_name] = param
            layer.update(layer_params)

class MLXModel(nn.Module):
  def __init__(self, layers):
    super().__init__()

    self.layers = layers

  def __call__(self, x):
    for m in self.layers:
      x = m(x)
    return x
