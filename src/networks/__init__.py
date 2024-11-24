import os

BACKEND = os.environ.get('RLTRADER_BACKEND', 'torch')
if BACKEND == 'torch':
    print("Using PyTorch as Backend")
    from .torch.cnn import CNN
    from .torch.dnn import DNN
    from .torch.lstm import LSTMNetwork
    from .torch.network import Network
else:
    raise ValueError(f"{BACKEND} is not supported")

__all__ = [
    'Network', 'DNN', 'LSTMNetwork', 'CNN'
]
