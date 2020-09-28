import math


def argmax(arr):
  return arr.index(max(arr))


class Correlation_Leveraging_UCB():
  def __init__(self, rho, tau, counts, q_values):
    self.rho = rho
    self.tau = tau
    self.counts = counts  # no. of times each arm has been pulled
    self.q_values = q_values  # current Q-value of each arm

  def initialize(self, n_arms):
    self.counts = [0]*n_arms
    self.q_values = [0.0]*n_arms

  def get_name(self):
    """Returns the name of this algorithm"""
    return "Corr.-Leveraging UCB"

  def get_corr_leverage(self, ucb_values, arm, n_arms):
    cum_Qs = 0  # Cumulative look-back/ahead Q-values
    for i in range(arm - self.tau, arm + self.tau):
      cum_Qs += (ucb_values[i] if i != arm else 0)
    corr_leverage = self.rho*(cum_Qs/(2*self.tau))
    return corr_leverage

  def select_arm(self):
    n_arms = len(self.counts)
    for arm in range(n_arms):
      if self.counts[arm] == 0:
        return arm  # need to select every at least once before applying UCB1 formula
    ucb_values = [0.0 for arm in range(n_arms)]
    for arm in range(n_arms):
      confidence_interval = math.sqrt((2*math.log(sum(self.counts)))/self.counts[arm])
      corr_leverage = self.get_corr_leverage(ucb_values, arm, n_arms)
      ucb_values[arm] = self.q_values[arm] + confidence_interval + corr_leverage
    return argmax(ucb_values)

  def update(self, chosen_arm, latest_reward):
    self.counts[chosen_arm] += 1
    N_t = self.counts[chosen_arm]  # already incremented to latest count
    prev_q = self.q_values[chosen_arm]
    new_empirical_mean = ((prev_q*(N_t - 1)) + latest_reward)/N_t
    self.q_values[chosen_arm] = new_empirical_mean
