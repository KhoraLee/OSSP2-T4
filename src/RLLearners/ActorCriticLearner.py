from .RLLearner import ReinforcementLearner
import numpy as np
from Networks import Network, DNN, LSTMNetwork, CNN
import utils

class ActorCriticLearner(ReinforcementLearner):
  def __init__(self, *args, shared_network=None,
               value_network_path=None, policy_network_path=None, **kwargs):
    super().__init__(*args, **kwargs)
    if shared_network is None:
      self.shared_network = Network.get_shared_network(
        net=self.net, num_steps=self.num_steps,
        input_dim=self.num_features,
        output_dim=self.agent.NUM_ACTIONS)
    else:
      self.shared_network = shared_network
    self.value_network_path = value_network_path
    self.policy_network_path = policy_network_path
    if self.value_network is None:
      self.init_value_network(shared_network=self.shared_network)
    if self.policy_network is None:
      self.init_policy_network(shared_network=self.shared_network)

  def get_batch(self):
    memory = zip(
      reversed(self.memory_sample),
      reversed(self.memory_action),
      reversed(self.memory_value),
      reversed(self.memory_policy),
      reversed(self.memory_reward),
    )
    x = np.zeros((len(self.memory_sample), self.num_steps, self.num_features))
    y_value = np.zeros((len(self.memory_sample), self.agent.NUM_ACTIONS))
    y_policy = np.zeros((len(self.memory_sample), self.agent.NUM_ACTIONS))
    value_max_next = 0
    for i, (sample, action, value, policy, reward) in enumerate(memory):
      x[i] = sample
      r = self.memory_reward[-1] - reward
      y_value[i, :] = value
      y_value[i, action] = r + self.discount_factor * value_max_next
      y_policy[i, :] = policy
      y_policy[i, action] = utils.sigmoid(r)
      value_max_next = value.max()
    return x, y_value, y_policy
