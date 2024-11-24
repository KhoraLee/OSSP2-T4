import mlx.core as mx
import mlx.nn as nn
import numpy as np
from .network import Network

class LSTMNetwork(Network):
  def __init__(self, *args, num_steps=1, **kwargs):
    self.num_steps = num_steps
    super().__init__(*args, **kwargs)

  @staticmethod
  def get_network_head(inp, output_dim):
    return [
      nn.BatchNorm(inp[1]),
      LSTMModule(inp[1], 128, batch_first=True, use_last_only=True),
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
    x = np.array(x).reshape((-1, self.num_steps, self.input_dim))
    return super().train_on_batch(x, y)

  def predict(self, sample):
    sample = np.array(sample).reshape((-1, self.num_steps, self.input_dim))
    return super().predict(sample)

class LSTMModule(nn.Module):
  def __init__(self, input_size, hidden_size, batch_first=True, use_last_only=False):
    super().__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.batch_first = batch_first
    self.use_last_only = use_last_only

    # LSTM 게이트를 위한 가중치 정의
    self.w_ih = nn.Linear(input_size, 4 * hidden_size, bias=True)
    self.w_hh = nn.Linear(hidden_size, 4 * hidden_size, bias=True)

  def __call__(self, x):
    # 입력 차원 처리
    if self.batch_first:
      batch_size, seq_len, _ = x.shape
    else:
      seq_len, batch_size, _ = x.shape
      x = mx.transpose(x, (1, 0, 2))

    h_t = mx.zeros((batch_size, self.hidden_size))
    c_t = mx.zeros((batch_size, self.hidden_size))
    outputs = []

    # LSTM 연산 수행
    for t in range(seq_len):
      x_t = x[:, t, :]

      # 게이트 계산
      gates = self.w_ih(x_t) + self.w_hh(h_t)
      i_t, f_t, g_t, o_t = mx.split(gates, 4, axis=1)

      # 활성화 함수 적용
      i_t = mx.sigmoid(i_t)
      f_t = mx.sigmoid(f_t)
      g_t = mx.tanh(g_t)
      o_t = mx.sigmoid(o_t)

      # 셀 상태와 은닉 상태 업데이트
      c_t = f_t * c_t + i_t * g_t
      h_t = o_t * mx.tanh(c_t)

      outputs.append(h_t)

    if self.use_last_only:
      return h_t

    outputs = mx.stack(outputs, axis=1)
    if not self.batch_first:
      outputs = mx.transpose(outputs, (1, 0, 2))

    return outputs