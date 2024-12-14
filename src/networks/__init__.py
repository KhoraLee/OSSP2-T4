import os

BACKEND = os.environ.get('RLTRADER_BACKEND', 'torch')
if BACKEND == 'torch':
    print("Using PyTorch as Backend")
    from .torch.cnn import CNN
    from .torch.dnn import DNN
    from .torch.lstm import LSTMNetwork
    from .torch.network import Network
    from .torch.cnn_lstm import CNN_LSTMNetwork
elif BACKEND == 'mlx':
    print("Using MLX as Backend")
    from .mlx.cnn import CNN
    from .mlx.dnn import DNN
    from .mlx.lstm import LSTMNetwork
    from .mlx.cnn_lstm import CNN_LSTMNetwork
    from .mlx.network import Network
else:
    raise ValueError(f"{BACKEND} is not supported")

__all__ = [
    'Network', 'DNN', 'LSTMNetwork', 'CNN', 'CNN_LSTMNetwork'
]
