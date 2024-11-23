from . import ActorCriticLearner
import numpy as np
import utils

class A2CLearner(ActorCriticLearner):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

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
    reward_next = self.memory_reward[-1]
    for i, (sample, action, value, policy, reward) in enumerate(memory):
      x[i] = sample
      r = reward_next + self.memory_reward[-1] - reward * 2
      reward_next = reward
      y_value[i, :] = value
      y_value[i, action] = np.tanh(r + self.discount_factor * value_max_next)
      advantage = y_value[i, action] - y_value[i].mean()
      y_policy[i, :] = policy
      y_policy[i, action] = utils.sigmoid(advantage)
      value_max_next = value.max()
    return x, y_value, y_policy
