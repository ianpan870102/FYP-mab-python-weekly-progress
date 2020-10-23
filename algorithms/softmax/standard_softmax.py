import math
import random


def weighted_draw(probs):
    r = random.random()  # uniformly random between 0 and 1
    cumulative_prob = 0
    for i, prob in enumerate(probs):
        cumulative_prob += prob
        if cumulative_prob > r:
            return i
    return len(probs) - 1  # last index


class Softmax:
    def __init__(self, n_arms, param_dict):
        self.tau = param_dict["tau"]
        self.counts = [0 for col in range(n_arms)]
        self.q_values = [0.0 for col in range(n_arms)]

    def get_name(self):
        """Returns the name of this algorithm"""
        return "Boltzmann (Softmax)"

    def select_arm(self):
        denom = sum([math.exp(q/self.tau) for q in self.q_values])
        probs = [math.exp(q/self.tau)/denom for q in self.q_values]
        return weighted_draw(probs)

    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] = self.counts[chosen_arm] + 1
        n = self.counts[chosen_arm]
        value = self.q_values[chosen_arm]
        # Smart way of
        new_value = ((n - 1)/n)*value + (1/float(n))*reward
        self.q_values[chosen_arm] = new_value
