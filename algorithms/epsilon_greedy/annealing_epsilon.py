import random
import math


def argmax(arr):
  return arr.index(max(arr))


class AnnealingEpsilonGreedy():
  def __init__(self, counts, q_values):
    self.counts = counts  # no. of times each arm has been pulled
    self.q_values = q_values  # current Q-value of each arm

  def initialize(self, n_arms):
    self.counts = [0]*n_arms
    self.q_values = [0.0]*n_arms

  def get_name(self):
    """Returns the name of this algorithm"""
    return "Annealing \u03F5-greedy"

  def select_arm(self):
    epsilon = 1/math.log((sum(self.counts) + 1) + 0.0000001)
    if random.random() > epsilon:  # exploit
      return argmax(self.q_values)
    else:  # explore
      return random.randrange(len(self.q_values))

  def update(self, chosen_arm, latest_reward):
    self.counts[chosen_arm] += 1
    N_t = self.counts[chosen_arm]  # already incremented to latest count
    prev_q = self.q_values[chosen_arm]
    new_empirical_mean = ((prev_q*(N_t - 1)) + latest_reward)/N_t
    self.q_values[chosen_arm] = new_empirical_mean
