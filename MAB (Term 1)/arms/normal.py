import random


class NormalArm():
  def __init__(self, mu, sigma):
    self.mu = mu
    self.sigma = sigma

  def draw_reward(self):
    """Draw reward based on normal/gaussian distribution"""
    return random.gauss(self.mu, self.sigma)
