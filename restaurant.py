#!/usr/bin/env python3
import random
import matplotlib.pyplot as plt
from arms.bernoulli import BernoulliArm
from arms.normal import NormalArm
from algorithms.epsilon_greedy.standard_epsilon import EpsilonGreedy
from algorithms.epsilon_greedy.annealing_epsilon import AnnealingEpsilonGreedy
from algorithms.softmax.standard_softmax import Softmax
from algorithms.softmax.annealing_softmax import AnnealingSoftmax
from algorithms.ucb.ucb1 import UCB1
from algorithms.ucb.ucb_bayesian import UCB_Bayesian
from algorithms.exp3.exp3 import Exp3
from algorithms.thompson.thompson import ThompsonSampling

import numpy as np

from matplotlib import rcParams
rcParams['font.family'] = ['Roboto']
for w in ["font.weight", "axes.labelweight", "axes.titleweight", "figure.titleweight"]:
    rcParams[w] = 'regular'


def argmax(arr):
    return arr.index(max(arr))


def plot_regret(X, Y, cumulative_optimal_reward, cumulative_reward, average_reward_in_each_round_1D, T, max_regret,
                min_regret, max_avg_reward, min_avg_reward, algo_name):
    fig, axs = plt.subplots(2)  # get two figures, top is regret, bottom is average reward in each round
    fig.suptitle(f'Graduate Students vs. Restaurants (Ch. 3 Exercise)\n{algo_name}')
    fig.subplots_adjust(hspace=0.5)

    axs[0].plot(X, Y, color='red', label='Regret of UCB')
    axs[0].set(xlabel='night', ylabel='Regret')
    axs[0].grid(True)
    axs[0].legend(loc='lower right')
    axs[0].set_xlim(0, T)

    axs[0].set_ylim(min_regret, 1.1*max_regret)
    axs[1].plot(X, average_reward_in_each_round_1D, color='black', label='average reward')

    axs[1].set(xlabel='nights', ylabel='Average Reward per night')
    axs[1].grid(True)
    axs[1].legend(loc='lower right')
    axs[1].set_xlim(0, T)
    # axs[1].set_ylim(min_avg_reward, 1.1*max_avg_reward)
    # Hard-coded for a smoother graph
    axs[1].set_ylim(0, 1.2)
    plt.savefig("./figures/restaurant_plot.png")
    plt.show()
    print("Thank you and good-bye.")


def plot_graph(timesteps, arms, algorithms, average_reward_in_each_round, average_cum_rewards, algorithm_arm_selections,
               algorithm_average_arm_selections, max_avg_reward, min_avg_reward, max_cum_reward, N, T, restaurant_names):
    """Plot a 3-row k-column graph with rolling average, cumulative reward and arm selection scatter plot."""

    num_of_algo = len(algorithms)

    plt.figure(figsize=(15, 10))
    for j, algo in enumerate(algorithms):
        plt.subplot(3, num_of_algo, j + 1)
        plt.plot(average_reward_in_each_round[j], label='Average rewards', color='palevioletred', alpha=0.8)
        plt.xlabel('Time-step t', fontsize=12)
        plt.ylabel(f'Avg. reward for {algo.get_name()}', fontsize=12)
        plt.title(f"{algo.get_name()}\nRolling Average Reward", fontsize=13)
        plt.axis([0, timesteps, min_avg_reward - 0.05, max_avg_reward + 0.05])

        plt.subplot(3, num_of_algo, j + 1 + num_of_algo)
        # DONE: As average_cum_rewards is not correct, graph values are not increasing
        plt.plot(average_cum_rewards[j], label='Cumulative reward', color='orange', alpha=0.8)
        plt.xlabel('Time-step t', fontsize=12)
        plt.ylabel(f'Cum. reward for {algo.get_name()}', fontsize=12)
        plt.title(f"Cumulative Reward", fontsize=13)
        plt.axis([0, timesteps, 0, max_cum_reward])

        plt.subplot(3, num_of_algo, j + 1 + 2*num_of_algo)
        # DONE: Plot scatter plot with transparency (using algorithm_average_arm_selections,
        # which is length T array composed of smaller length N (no. of arms) arrays that sum up to 1.0)

        cm = plt.cm.get_cmap('YlOrBr')
        for n in range(N):
            plt.scatter(range(timesteps), [n + 1]*T, s=2.5, c=algorithm_average_arm_selections[j][n], alpha=.3, cmap=cm)

        plt.axis([0, timesteps, 0, len(arms) + 1])
        plt.xlabel('Time-step t', fontsize=12)
        plt.ylabel(f'{algo.get_name()}\'s arm selection', fontsize=12)
        plt.yticks(ticks=arms, labels=restaurant_names)
        plt.title("Arm selection scatter plot", fontsize=13)

    plt.tight_layout(pad=2.0)
    # plt.savefig('./figures/graph_demo.png')
    plt.show()


def plot_cum_rewards(algorithms, average_cum_rewards, timesteps, max_cum_reward):
    """Cumulative rewards of k algorithms in 1 line graph."""
    plt.figure()
    for i in range(len(algorithms)):
        plt.plot(average_cum_rewards[i], label='Cumulative reward', alpha=0.8)
    plt.legend([algo.get_name() for algo in algorithms])
    plt.xlabel('Time-step t', fontsize=12)
    plt.ylabel(f'Cum. reward comparisons', fontsize=12)
    plt.title(f"Cumulative Reward", fontsize=13)
    plt.axis([0, timesteps, 0, max_cum_reward])
    # plt.savefig('./figures/cum_rewards.png')
    plt.show()


def get_inputs():
    N = int(input("Please enter the number of restaurants near your living quarter: "))
    T = int(input("Please enter the number of nights you will be eating out: "))

    restaurant_names = []
    for i in range(N):
        restaurant_names.append(input(f"Enter restaurant #{i+1} name: "))

    ans = input("Do you have the average ratings of these restaurants (Google Maps, OpenRice)? [y/n]: ")
    flag = True if ans.lower() == 'y' else False

    ratings = []  # act as mu later for arms
    if flag:
        for i in range(N):
            ratings.append(float(input(f"Please rate {restaurant_names[i]} between 0 and 1: ")))
    else:
        print("That's fine, we'll randomly initialise the restaurant ratings for you.")

    return N, T, flag, restaurant_names, ratings


def main():
    N, T, flag, restaurant_names, ratings = get_inputs()

    arms = []

    for i in range(N):
        mu = random.random() if not flag else ratings[i]
        arms.append(NormalArm(mu, 0.5))

    max_mu = max([arm.mu for arm in arms])
    n_arms = len(arms)
    optimal_index = argmax([arm.mu for arm in arms])
    print(f'Best restaurant: {restaurant_names[optimal_index]} (#{optimal_index + 1})')
    print(f"Their mu's are {arms[0].mu}, {arms[1].mu}, and {arms[2].mu}")

    param_dict = {"epsilon": 0.5, "sigma": 1.96, "tau": 0.2, "gamma": 0.2}
    algo_epsilon = EpsilonGreedy(n_arms, param_dict)
    algo_anneal_epsilon = AnnealingEpsilonGreedy(n_arms, param_dict)
    algo_ucb1 = UCB1(n_arms, param_dict)
    algo_ucb_bayesian = UCB_Bayesian(n_arms, param_dict)  # 95% confident
    algo_softmax = Softmax(n_arms, param_dict)
    algo_anneal_softmax = AnnealingSoftmax(n_arms, param_dict)
    algo_exp3 = Exp3(n_arms, param_dict)
    algo_thompson = ThompsonSampling(n_arms, param_dict)

    algorithms = [algo_ucb1, algo_epsilon, algo_softmax]

    # semi-global variables
    timesteps = T
    total_iteration = 100  # outer-loop

    # 3D list[algo][t] (array of running avg. rewards for each algo at time-step t of outer loop i) (will eventually become 3D)
    algorithm_rewards = np.zeros(shape=(len(algorithms), total_iteration, T), dtype=float)

    # 3D list[algo][t] (array of cumulative rewards for each algo at time-step t of outer loop i)
    algorithm_cum_rewards = np.zeros(shape=(len(algorithms), total_iteration, T), dtype=float)

    # 3D list[algo][i][t] (array of arm selections (indices) for each algo at time-step t of outer loop i)
    algorithm_arm_selections = np.zeros(shape=(len(algorithms), total_iteration, T), dtype=float)

    algorithm_timestep_reward_stacked = np.zeros(shape=(len(algorithms), T), dtype=float)

    arm_selection_count = np.zeros(shape=(len(algorithms), T, N), dtype=float)

    average_reward_in_each_round = np.zeros(shape=(len(algorithms), T), dtype=float)

    algorithm_average_arm_selections = np.zeros(shape=(len(algorithms), N, T), dtype=float)

    algorithm_cum_rewards_stacked = np.zeros(shape=(len(algorithms), T), dtype=float)

    average_cum_rewards = np.zeros(shape=(len(algorithms), T), dtype=float)

    for j, algo in enumerate(algorithms):

        for i in range(total_iteration):  # one graduate student, no need to average preference
            avg_rewards = np.zeros(shape=(T), dtype=float)
            cum_rewards = np.zeros(shape=(T), dtype=float)
            arm_selections = np.zeros(shape=(T))
            new_avg = 0
            # DONE: reinitialize algo dynamically based on algo type
            algo.__init__(n_arms, param_dict)  # reinitialization (prevent regret going down)
            for t in range(timesteps):  # NOTE: 0 based? 1 based?
                chosen_arm = algo.select_arm()

                arm_selections[t] = chosen_arm + 1

                reward = arms[chosen_arm].draw_reward()
                algorithm_timestep_reward_stacked[j][t] += reward  # This persists over total_iterations

                # DONE: create a list to keep track of arm selected (count)
                arm_selection_count[j][t][chosen_arm] += 1

                new_avg = (cum_rewards[t - 1] + reward)/(t + 1) if t > 0 else reward  # new running avg.

                avg_rewards[t] = new_avg
                cum_rewards[t] = cum_rewards[t - 1] + reward if t > 0 else reward

                # doesn't clear memory, persists through total_iteration
                algorithm_cum_rewards_stacked[j][t] += cum_rewards[t]
                # if j == 0:
                #     print(f"{new_avg}\t{t}\t{cum_rewards[t]}")

                algo.update(chosen_arm, reward)

            algorithm_arm_selections[j][i] = arm_selections  # 1D by itself
            algorithm_rewards[j][i] = avg_rewards
            algorithm_cum_rewards[j][i] = cum_rewards

        cumulative_optimal_reward = 0.0
        cumulative_reward = 0.0
        x_axis = np.zeros(T, dtype=int)
        regrets = np.zeros(T, dtype=float)  # regret for each round

        for t in range(T):
            average_reward_in_each_round[j][t] = float(algorithm_timestep_reward_stacked[j][t])/float(total_iteration)
            average_cum_rewards[j][t] = float(algorithm_cum_rewards_stacked[j][t])/float(total_iteration)

        for t in range(T):
            for arm_index in range(N):
                algorithm_average_arm_selections[j][arm_index][t] = float(
                    arm_selection_count[j][t][arm_index])/float(total_iteration)

        for t in range(timesteps):
            x_axis[t] = t
            cumulative_optimal_reward += max_mu
            cumulative_reward += average_reward_in_each_round[j][t]
            # The cumulative doesn't have to be strictly increasing as there is a chance the cumulative_reward is better
            # than the cumulative_optimal_reward (due to standard deviation)
            regrets[t] = max(0, cumulative_optimal_reward - cumulative_reward)

        # DONE: Redefine plot_regret() to handle 3D
        plot_regret(x_axis, regrets, cumulative_optimal_reward, cumulative_reward, average_reward_in_each_round[j],
                    timesteps, max(regrets), min(regrets), max(average_reward_in_each_round[j]),
                    min(average_reward_in_each_round[j]), algo.get_name())
        print(f"The average regret for {algo.get_name()} is {cumulative_optimal_reward - cumulative_reward}")

    max_avg_reward, min_avg_reward, max_cum_reward = 0, 0, 0
    for j in range(len(algorithms)):
        max_cum_reward = max(max_cum_reward, average_cum_rewards[j][-1])
        for t in range(T):
            min_avg_reward = min(min_avg_reward, average_reward_in_each_round[j][t])
            max_avg_reward = max(max_avg_reward, average_reward_in_each_round[j][t])

    plot_graph(timesteps, arms, algorithms, average_reward_in_each_round, average_cum_rewards, algorithm_arm_selections,
               algorithm_average_arm_selections, max_avg_reward, min_avg_reward, max_cum_reward, N, T, restaurant_names)
    plot_cum_rewards(algorithms, average_cum_rewards, timesteps, max_cum_reward)


if __name__ == "__main__":
    main()
