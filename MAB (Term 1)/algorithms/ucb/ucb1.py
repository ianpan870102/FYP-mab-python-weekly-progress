import math


def argmax(arr):
    return arr.index(max(arr))


class UCB1():
    def __init__(self, n_arms, _):
        self.counts = [0]*n_arms  # no. of times each arm has been pulled
        self.q_values = [0.0]*n_arms  # current Q-value of each arm
        self.ucb_values = [0.0 for arm in range(n_arms)]

    def get_name(self):
        """Returns the name of this algorithm"""
        return "UCB1"

    def select_arm(self):
        n_arms = len(self.counts)
        for arm in range(n_arms):
            if self.counts[arm] == 0:
                return arm  # need to select every arm at least once before applying UCB1 formula
        # ucb_values = [0.0 for arm in range(n_arms)]
        for arm in range(n_arms):
            confidence_interval = math.sqrt((2*math.log(sum(self.counts)))/float(self.counts[arm]))
            self.ucb_values[arm] = self.q_values[arm] + confidence_interval
        return argmax(self.ucb_values)

    def update(self, chosen_arm, latest_reward):
        self.counts[chosen_arm] += 1
        N_t = self.counts[chosen_arm]  # already incremented to latest count
        prev_q = self.q_values[chosen_arm]
        new_empirical_mean = ((prev_q*(N_t - 1)) + latest_reward)/N_t
        self.q_values[chosen_arm] = new_empirical_mean
