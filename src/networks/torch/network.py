import abc
import threading
import torch
from ..base_network import BaseNetwork
device = torch.device('cuda' if torch.cuda.is_available() else
                      # 'mps' if torch.mps.is_available() else # currently CPU is faster than MPS
                      'cpu')
print(f'Pytorch Backend: {device}')

class Network(BaseNetwork):
    lock = threading.Lock()

    def __init__(self, input_dim=0, output_dim=0, lr=0.001, shared_network=None, activation='relu', loss='mse'):
        super().__init__(input_dim, output_dim, lr, shared_network, activation, loss)

        self.model = torch.nn.Sequential(self.head)
        if self.activation == 'linear':
            pass
        elif self.activation == 'relu':
            self.model.add_module('activation', torch.nn.ReLU())
        elif self.activation == 'leaky_relu':
            self.model.add_module('activation', torch.nn.LeakyReLU())
        elif self.activation == 'sigmoid':
            self.model.add_module('activation', torch.nn.Sigmoid())
        elif self.activation == 'tanh':
            self.model.add_module('activation', torch.nn.Tanh())
        elif self.activation == 'softmax':
            self.model.add_module('activation', torch.nn.Softmax(dim=1))
        self.model.apply(Network.init_weights)
        self.model.to(device)

        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.lr)
        # self.optimizer = torch.optim.NAdam(self.model.parameters(), lr=self.lr)
        self.criterion = None
        if loss == 'mse':
            self.criterion = torch.nn.MSELoss()
        elif loss == 'binary_crossentropy':
            self.criterion = torch.nn.BCELoss()

    def predict(self, sample):
        with self.lock:
            self.model.eval()
            with torch.no_grad():
                x = torch.from_numpy(sample).float().to(device)
                pred = self.model(x).detach().cpu().numpy()
                pred = pred.flatten()
            return pred

    def train_on_batch(self, x, y):
        loss = 0.
        with self.lock:
            self.model.train()
            x = torch.from_numpy(x).float().to(device)
            y = torch.from_numpy(y).float().to(device)
            y_pred = self.model(x)
            _loss = self.criterion(y_pred, y)
            self.optimizer.zero_grad()
            _loss.backward()
            self.optimizer.step()
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

    @abc.abstractmethod
    def get_network_head(inp, output_dim):
        pass

    @staticmethod
    def init_weights(m):
        if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv1d):
            torch.nn.init.normal_(m.weight, std=0.01)
        elif isinstance(m, torch.nn.LSTM):
            for weights in m.all_weights:
                for weight in weights:
                    torch.nn.init.normal_(weight, std=0.01)

    def save_model(self, model_path):
        if model_path is not None and self.model is not None:
            torch.save(self.model, model_path)

    def load_model(self, model_path):
        if model_path is not None:
            self.model = torch.load(model_path)
