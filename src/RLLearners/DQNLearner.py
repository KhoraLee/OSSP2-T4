from .RLLearner import ReinforcementLearner
import numpy as np

class DQNLearner(ReinforcementLearner):
  def __init__(self, *args, value_network_path=None, **kwargs):
    super().__init__(*args, **kwargs)
    self.value_network_path = value_network_path
    self.init_value_network()

  def get_batch(self):
    memory = zip(
      reversed(self.memory_sample),
      reversed(self.memory_action),
      reversed(self.memory_value),
      reversed(self.memory_reward),
    )
    x = np.zeros((len(self.memory_sample), self.num_steps, self.num_features))
    y_value = np.zeros((len(self.memory_sample), self.agent.NUM_ACTIONS))
    value_max_next = 0
    for i, (sample, action, value, reward) in enumerate(memory):
      x[i] = sample
      r = self.memory_reward[-1] - reward
      y_value[i] = value
      y_value[i, action] = r + self.discount_factor * value_max_next
      value_max_next = value.max()
    return x, y_value, None
