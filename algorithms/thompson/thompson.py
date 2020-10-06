def argmax(arr):
  return arr.index(max(arr))


class ThompsonSampling():
  def __init__(self):
    pass

  def get_name(self):
    """Returns the name of this algorithm"""
    return "Thompson Sampling"

  def select_arm(self) -> int:
    """Returns the chosen arm's index (0-based)."""
    pass

  def update(self, chosen_arm, latest_reward):
    pass
