import random
import math


# probs: [0.1, 0.1, 0.6, 0.2]
def weighted_draw(probs):
    r = random.random()  # uniformly random between 0 and 1
    cumulative_prob = 0
    for i, prob in enumerate(probs):
        cumulative_prob += prob
        if cumulative_prob > r:
            return i
    return len(probs) - 1  # last index


class Exp3():
    def __init__(self, n_arms, param_dict):
        self.gamma = param_dict["gamma"]
        self.weights = [1.0]*n_arms  # equally weighted

    def get_name(self):
        """Returns the name of this algorithm"""
        return "Exp3"

    def select_arm(self):
        n_arms = len(self.weights)
        total_weight = sum(self.weights)
        probs = [(1 - self.gamma)*(w_i/total_weight) + self.gamma/n_arms for w_i in self.weights]
        return weighted_draw(probs)

    def update(self, chosen_arm, latest_reward):
        n_arms = len(self.weights)
        total_weight = sum(self.weights)
        probs = [(1 - self.gamma)*(w_i/total_weight) + self.gamma/n_arms for w_i in self.weights]
        x_hat = latest_reward/probs[chosen_arm]  # estimated reward
        self.weights[chosen_arm] *= math.exp(x_hat*(self.gamma/n_arms))  # all other weights stay the same
