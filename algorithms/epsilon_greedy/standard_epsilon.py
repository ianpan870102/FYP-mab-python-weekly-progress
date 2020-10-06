import random


def argmax(arr):
  return arr.index(max(arr))


class EpsilonGreedy():
  def __init__(self, epsilon: float, n_arms):
    self.epsilon = epsilon
    self.counts = [0]*n_arms  # no. of times each arm has been pulled
    self.q_values = [0.0]*n_arms  # current Q-value of each arm

  def get_name(self):
    """Returns the name of this algorithm"""
    return "\u03F5-greedy"

  def select_arm(self) -> int:
    """Returns the chosen arm's index (0-based)."""
    if random.random() > self.epsilon:  # exploit
      return argmax(self.q_values)  # most optimal arm
    else:  # explore
      return random.randrange(len(self.q_values))

  def update(self, chosen_arm, latest_reward):
    self.counts[chosen_arm] += 1
    N_t = self.counts[chosen_arm]  # already incremented to latest count
    prev_q = self.q_values[chosen_arm]
    new_empirical_mean = ((prev_q*(N_t - 1)) + latest_reward)/N_t
    self.q_values[chosen_arm] = new_empirical_mean
