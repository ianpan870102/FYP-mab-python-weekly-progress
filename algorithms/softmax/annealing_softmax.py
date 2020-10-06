import math
import random


def weighted_draw(probs):
  z = random.random()  # between 0 and 1
  cumulative_prob = 0.0
  for i, prob in enumerate(probs):
    cumulative_prob += prob
    if cumulative_prob > z:
      return i
  return len(probs) - 1  # last index


class AnnealingSoftmax:
  def __init__(self, counts, values):
    self.counts = counts
    self.values = values

  def initialize(self, n_arms):
    self.counts = [0 for col in range(n_arms)]
    self.values = [0.0 for col in range(n_arms)]

  def get_name(self):
    """Returns the name of this algorithm"""
    return "Annealing Boltzmann"

  def select_arm(self):
    t = sum(self.counts) + 1
    tau = 1/math.log(t + 0.0000001)
    z = sum([math.exp(v/tau) for v in self.values])
    probs = [math.exp(v/tau)/z for v in self.values]
    return weighted_draw(probs)

  def update(self, chosen_arm, reward):
    self.counts[chosen_arm] = self.counts[chosen_arm] + 1
    n = self.counts[chosen_arm]
    value = self.values[chosen_arm]
    new_value = ((n - 1)/float(n))*value + (1/float(n))*reward
    self.values[chosen_arm] = new_value
