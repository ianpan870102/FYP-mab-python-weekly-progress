# This algorithm deals solely with Bernoulli arm rewards
import random


def argmax(arr):
  return arr.index(max(arr))


class ThompsonSampling():
  def __init__(self, n_arms):
    self.n_arms = n_arms
    self.reward_equal_to_1 = [0]*n_arms
    self.reward_equal_to_0 = [0]*n_arms
    self.cum_reward = 0

  def get_name(self):
    """Returns the name of this algorithm"""
    return "Thompson Sampling"

  def select_arm(self) -> int:
    """Returns the chosen arm's index (0-based)."""
    random_beta_list = [
        random.betavariate(self.reward_equal_to_1[i] + 1, self.reward_equal_to_0[i] + 1) for i in range(self.n_arms)
    ]
    return argmax(random_beta_list)

  def update(self, chosen_arm, latest_reward):
    if latest_reward == 1:
      self.reward_equal_to_1[chosen_arm] += 1
    else:
      self.reward_equal_to_0[chosen_arm] += 1
    self.cum_reward += latest_reward
