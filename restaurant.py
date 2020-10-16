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


def plot_regret(X, Y, cumulative_optimal_reward, cumulative_reward, average_reward_in_each_round, T, max_regret,
                min_regret, max_avg_reward, min_avg_reward):
    fig, axs = plt.subplots(2)  # get two figures, top is regret, bottom is average reward in each round
    fig.suptitle('Graduate Students vs. Restaurants (Ch. 3 Exercise)')
    fig.subplots_adjust(hspace=0.5)

    axs[0].plot(X, Y, color='red', label='Regret of UCB')
    axs[0].set(xlabel='night', ylabel='Regret')
    axs[0].grid(True)
    axs[0].legend(loc='lower right')
    axs[0].set_xlim(0, T)

    axs[0].set_ylim(min_regret, 1.1*max_regret)
    axs[1].plot(X, average_reward_in_each_round, color='black', label='average reward')

    axs[1].set(xlabel='nights', ylabel='Average Reward per night')
    axs[1].grid(True)
    axs[1].legend(loc='lower right')
    axs[1].set_xlim(0, T)
    axs[1].set_ylim(min_avg_reward, 1.1*max_avg_reward)
    plt.savefig("./figures/restaurant_plot.png")
    plt.show()
    print("Thank you and good-bye.")


def plot_graph(timesteps, arms, algorithms, algorithm_rewards, algorithm_cum_rewards, algorithm_arm_selections, max_mu,
               max_cum_reward):
    """Plot a 3-row k-column graph with rolling average, cumulative reward and arm selection scatter plot."""
    num_of_algo = len(algorithms)

    plt.figure(figsize=(15, 10))
    for i, algo in enumerate(algorithms):
        plt.subplot(3, num_of_algo, i + 1)
        plt.plot(algorithm_rewards[i], label='Average rewards', color='palevioletred', alpha=0.8)
        plt.xlabel('Time-step t', fontsize=12)
        plt.ylabel(f'Avg. reward for {algo.get_name()}', fontsize=12)
        plt.title(f"{algo.get_name()}\nRolling Average Reward", fontsize=13)
        plt.axis([0, timesteps, 0, max_mu + 0.1])

        plt.subplot(3, num_of_algo, i + 1 + num_of_algo)
        plt.plot(algorithm_cum_rewards[i], label='Cumulative reward', color='orange', alpha=0.8)
        plt.xlabel('Time-step t', fontsize=12)
        plt.ylabel(f'Cum. reward for {algo.get_name()}', fontsize=12)
        plt.title(f"Cumulative Reward", fontsize=13)
        plt.axis([0, timesteps, 0, max_cum_reward])

        plt.subplot(3, num_of_algo, i + 1 + 2*num_of_algo)
        plt.scatter(range(timesteps), algorithm_arm_selections[i], s=2.5, color='salmon', alpha=.9)
        plt.axis([0, timesteps, 0, len(arms) + 1])
        plt.xlabel('Time-step t', fontsize=12)
        plt.ylabel(f'{algo.get_name()}\'s arm selection', fontsize=12)
        plt.title("Arm selection scatter plot", fontsize=13)

    plt.tight_layout(pad=2.0)
    # plt.savefig('./figures/graph_demo.png')
    plt.show()


def plot_cum_rewards(algorithms, algorithm_cum_rewards, timesteps, max_cum_reward):
    """Cumulative rewards of k algorithms in 1 line graph."""
    plt.figure()
    for i in range(len(algorithms)):
        plt.plot(algorithm_cum_rewards[i], label='Cumulative reward', alpha=0.8)
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

    algo_epsilon = EpsilonGreedy(0.05, n_arms)
    algo_anneal_epsilon = AnnealingEpsilonGreedy(n_arms)
    algo_ucb1 = UCB1(n_arms)
    algo_ucb_bayesian = UCB_Bayesian(1.96, n_arms)  # 95% confident
    algo_softmax = Softmax(.2, n_arms)
    algo_anneal_softmax = AnnealingSoftmax(n_arms)
    algo_exp3 = Exp3(.2, n_arms)
    algo_thompson = ThompsonSampling(n_arms)

    algorithms = [algo_ucb1]

    # semi-global variables
    timesteps = T
    total_iteration = 5  # outer-loop

    # 3D list[algo][t] (array of running avg. rewards for each algo at time-step t of outer loop i) (will eventually become 3D)
    # algorithm_rewards = [[]*len(algorithms)]
    algorithm_rewards = np.zeros(shape=(len(algorithms), total_iteration, T), dtype=float)

    # 3D list[algo][t] (array of cumulative rewards for each algo at time-step t of outer loop i)
    # algorithm_cum_rewards = [[]*len(algorithms)]
    algorithm_cum_rewards = np.zeros(shape=(len(algorithms), total_iteration, T), dtype=float)

    # 3D list[algo][i][t] (array of arm selections for each algo at time-step t of outer loop i)
    # algorithm_arm_selections = [[]*len(algorithms)]
    algorithm_arm_selections = np.zeros(shape=(len(algorithms), total_iteration, T), dtype=float)

    reward_round_iteration = np.zeros(shape=(len(algorithms), T), dtype=float)

    arm_selection_count = np.zeros(shape=(len(algorithms), N), dtype=float)

    average_reward_in_each_round = np.zeros(shape=(len(algorithms), T), dtype=float)

    algorithm_average_arm_selections = np.zeros(shape=(len(algorithms), N), dtype=float)

    for j, algo in enumerate(algorithms):

        for i in range(total_iteration):  # one graduate student, no need to average preference
            avg_rewards = np.zeros(shape=(T), dtype=float)
            cum_rewards = np.zeros(shape=(T), dtype=float)
            arm_selections = np.zeros(shape=(T))
            new_avg = 0
            # TODO: reinitialize algo dynamically based on algo type
            # algo = get_type(algo).reinit()  # pseudocode
            algo = UCB1(n_arms)  # reinitialize algorithm (clear previous memory)
            for t in range(timesteps):  # NOTE: 0 based? 1 based?
                chosen_arm = algo.select_arm()

                # arm_selections.append(chosen_arm + 1)  # convert 0-based index to 1-based
                arm_selections[t] = chosen_arm + 1

                reward = arms[chosen_arm].draw_reward()
                reward_round_iteration[j][t] += reward  # This persists over total_iterations

                # DOING: create a list to keep track of arm selected (count)
                arm_selection_count[j][chosen_arm] += 1

                new_avg = (avg_rewards[-1]*(t - 1) + reward)/t if t > 0 else 0  # new running avg.

                # avg_rewards.append(new_avg)  # records of running averages (up to time t)
                # cum_rewards.append(new_avg*t)  # records of cumulative rewards (up to time t)
                avg_rewards[t] = new_avg
                cum_rewards[t] = new_avg*t

                algo.update(chosen_arm, reward)

            # algorithm_arm_selections[j].append(arm_selections)  # NOTE: scatter plots can be overwritten
            # algorithm_rewards[j].append(avg_rewards)
            # algorithm_cum_rewards[j].append(cum_rewards)

            algorithm_arm_selections[j][i] = arm_selections
            algorithm_rewards[j][i] = avg_rewards
            algorithm_cum_rewards[j][i] = cum_rewards

        cumulative_optimal_reward = 0.0
        cumulative_reward = 0.0
        x_axis = np.zeros(timesteps, dtype=int)
        regrets = np.zeros(timesteps, dtype=float)  # regret for each round

        # print("Average reward in each round:", average_reward_in_each_round)

        for t in range(timesteps):
            average_reward_in_each_round[j][t] = float(reward_round_iteration[j][t])/float(total_iteration)

        print(f">>>>>> {arm_selection_count}")
        for arm_index in range(N):
            algorithm_average_arm_selections[j][arm_index] = float(
                arm_selection_count[j][arm_index])/float(total_iteration)

        for t in range(timesteps):
            x_axis[t] = t
            cumulative_optimal_reward += max_mu
            cumulative_reward += average_reward_in_each_round[j][t]
            regrets[t] = max(0, cumulative_optimal_reward - cumulative_reward)

        # TODO: Redefine plot_regret() to handle 3D
        plot_regret(x_axis, regrets, cumulative_optimal_reward, cumulative_reward, average_reward_in_each_round,
                    timesteps, max(regrets), min(regrets), max(average_reward_in_each_round),
                    min(average_reward_in_each_round))
        print(f"The average regret for {algo.get_name()} is {cumulative_optimal_reward - cumulative_reward}")

    max_cum_reward = 0
    for j in range(len(algorithms)):
        for i in range(total_iteration):
            max_cum_reward = max(max_cum_reward, algorithm_cum_rewards[j][i][-1])

    # Compute average rewards for each iteration, for each algorithm
    # average_reward_in_each_round = np.zeros(shape=(len(algorithms), T), dtype=float)
    # Calculate the values for one good 1000 rounds
    # Squash 200X1000 -> 1X1000
    # for j, algo in enumerate(algorithms):
    #     for t in range(timesteps):
    #         average_reward_in_each_round[j][t] = float(reward_round_iteration[j][t])/float(total_iteration)

    # Squash 3D algorithm_arm_selections from [algo][i][t] -> [algo][t]
    # 2D algorithm_average_arm_selections[j][t] -> for algorithm j, timestep t: [.2, .1, .5, .2] <- count of each arm / total_iterations
    # algorithm_average_arm_selections = np.zeros(shape=(len(algorithms), T), dtype=float)
    # for j, algo in enumerate(algorithms):
    #     for t in range(timesteps):
    #         algorithm_average_arm_selections[j][t] = float(arm_selection_count[j][t])/float(total_iteration)

    # TODO: Redefine plot_graph() to handle 3D
    # plot_graph(timesteps, arms, algorithms, algorithm_rewards, algorithm_cum_rewards, algorithm_average_arm_selections,
    #            max_mu, max_cum_reward)
    # plot_cum_rewards(algorithms, algorithm_cum_rewards, timesteps, max_cum_reward)


if __name__ == "__main__":
    main()
