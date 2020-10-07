# This is the 4th program to simulate the multi-arm bandit
# Let say we only use the  UCB
# Each arm has outcome 0 or 1, with probability 1 being the winning probability (Bernoulli distribution)

# Created by John C.S. Lui     Date: April 17, 2020

import numpy as np
from scipy.stats import bernoulli  # import bernoulli
import matplotlib.pyplot as plt

Num_of_Arms = 4  # number of arms

#  input parameters
winning_parameters = np.array([0.2, 0.3, 0.2, 0.9], dtype=float)
max_prob = 0.9  # record the highest probability of winning for all arms
optimal_arm = 3  # index for the optimal arm
T = 1000  # number of rounds to simulate
total_iteration = 10  # number of iterations to the MAB simulation

reward_round_iteration = np.zeros((T), dtype=int)  # reward in each round average by # of iteration

num_selected = np.zeros((Num_of_Arms), dtype=int)  # track # of times selected for each arm
cumulative_reward = np.zeros((Num_of_Arms), dtype=float)  # track cumulative reward for each arm
estimated_reward = np.zeros((Num_of_Arms), dtype=float)  # track estimate reward for each arm

# Go through T rounds, each round we need to select an arm

for iteration_count in range(total_iteration):
  for round in range(T):
    if round < Num_of_Arms:
      select_arm = round  # select round sequentially
    else:  # select the best estimated arm so far
      select_arm = np.argmax(estimated_reward)

    # generate reward and update the variables
    current_reward = bernoulli.rvs(winning_parameters[select_arm])
    num_selected[select_arm] += 1  # how many times each arm has been selected (equiv. to our self.counts)
    cumulative_reward[select_arm] += float(current_reward)
    # compute UCB estimate. Note that we need to do log(round +1) because round starts with 0
    estimated_reward[select_arm]  = cumulative_reward[select_arm]/float(num_selected[select_arm]) +  \
             np.sqrt(2*np.log(round+1)/float(num_selected[select_arm]))

    reward_round_iteration[round] += current_reward

  # after one iteration, need to reset variables
  num_selected = np.zeros((Num_of_Arms), dtype=int)
  cumulative_reward = np.zeros((Num_of_Arms), dtype=float)
  estimated_reward = np.zeros((Num_of_Arms), dtype=float)

# compute average reward for each iteration

average_reward_in_each_round = np.zeros(T, dtype=float)

# The max value in average_reward_round can only be 1 (eg:
for round in range(T):
  average_reward_in_each_round[round] = float(reward_round_iteration[round])/float(total_iteration)

# Let generate X and Y data points to plot it out

cumulative_optimal_reward = 0.0
cumulative_reward = 0.0

X = np.zeros(T, dtype=int)
Y = np.zeros(T, dtype=float)

print(average_reward_in_each_round)

# cumulative reward = (if you have a 0.9 chance of getting one, after 100 rounds is 100*0.9 = 90 )
for round in range(T):
  X[round] = round
  cumulative_optimal_reward += max_prob
  cumulative_reward += average_reward_in_each_round[round]
  # print(f"{cumulative_optimal_reward} \t {cumulative_reward}")
  Y[round] = cumulative_optimal_reward - cumulative_reward

#After 200 1000 rounds
print('After ', T, 'rounds, regret is: ', cumulative_optimal_reward - cumulative_reward)

fig, axs = plt.subplots(2)  # get two figures, top is regret, bottom is average reward in each round
fig.suptitle('John\'s Performance of UCB Arm Selection')
fig.subplots_adjust(hspace=0.5)

axs[0].plot(X, Y, color='red', label='Regret of UCB')
axs[0].set(xlabel='round number', ylabel='Regret')
axs[0].grid(True)
axs[0].legend(loc='lower right')
axs[0].set_xlim(0, T)
axs[0].set_ylim(0, 1.1*(cumulative_optimal_reward - cumulative_reward))

axs[1].plot(X, average_reward_in_each_round, color='black', label='average reward')
axs[1].set(xlabel='round number', ylabel='Average Reward per round')
axs[1].grid(True)
axs[1].legend(loc='lower right')
axs[1].set_xlim(0, T)
axs[1].set_ylim(0, 1.0)
# plt.savefig("prog3_figure.png")
plt.show()
