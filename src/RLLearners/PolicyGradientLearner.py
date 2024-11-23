from .RLLearner import ReinforcementLearner
import numpy as np
import utils

class PolicyGradientLearner(ReinforcementLearner):
  def __init__(self, *args, policy_network_path=None, **kwargs):
    super().__init__(*args, **kwargs)
    self.policy_network_path = policy_network_path
    self.init_policy_network()

  def get_batch(self):
    memory = zip(
      reversed(self.memory_sample),
      reversed(self.memory_action),
      reversed(self.memory_policy),
      reversed(self.memory_reward),
    )
    x = np.zeros((len(self.memory_sample), self.num_steps, self.num_features))
    y_policy = np.zeros((len(self.memory_sample), self.agent.NUM_ACTIONS))
    for i, (sample, action, policy, reward) in enumerate(memory):
      x[i] = sample
      r = self.memory_reward[-1] - reward
      y_policy[i, :] = policy
      y_policy[i, action] = utils.sigmoid(r)
    return x, None, y_policy
