import abc
import threading

class BaseNetwork(metaclass=abc.ABCMeta):
  lock = threading.Lock()

  def __init__(self, input_dim=0, output_dim=0, lr=0.001,
               shared_network=None, activation='relu', loss='mse'):
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.lr = lr
    self.shared_network = shared_network
    self.activation = activation
    self.loss = loss

    inp = None
    if hasattr(self, 'num_steps'):
      inp = (self.num_steps, input_dim)
    else:
      inp = (self.input_dim,)

    self.head = None
    if self.shared_network is None:
      self.head = self.get_network_head(inp, self.output_dim)
    else:
      self.head = self.shared_network

  @abc.abstractmethod
  def predict(self, sample):
    pass

  @abc.abstractmethod
  def train_on_batch(self, x, y):
    pass

  @abc.abstractmethod
  def get_shared_network(cls, net='dnn', num_steps=1, input_dim=0, output_dim=0):
    pass

  @staticmethod
  @abc.abstractmethod
  def get_network_head(inp, output_dim):
    pass

  @abc.abstractmethod
  def save_model(self, model_path):
    pass

  @abc.abstractmethod
  def load_model(self, model_path):
    pass