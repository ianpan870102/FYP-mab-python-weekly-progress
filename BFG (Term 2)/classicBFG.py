import random
import numpy as np
import matplotlib.pyplot as plt
# from arms.bernoulli import BernoulliArm
# from arms.normal import NormalArm

from matplotlib import rcParams
rcParams['font.family'] = ['Roboto']
for w in ["font.weight", "axes.labelweight", "axes.titleweight", "figure.titleweight"]:
    rcParams[w] = 'regular'


def plot_regret(X, Y, cumulative_optimal_reward, cumulative_reward, average_reward_in_each_round, T):
    fig, axs = plt.subplots(2)  # get two figures, top is regret, bottom is average reward in each round
    fig.suptitle(f'Performance of Classic BFG')
    fig.subplots_adjust(hspace=0.5)

    axs[0].plot(X, Y, color='red', label='Regret of UCB')
    axs[0].set(xlabel='round number', ylabel='Regret')
    axs[0].grid(True)
    axs[0].legend(loc='lower right')
    axs[0].set_xlim(0, T)
    axs[0].set_ylim(0, 1.1*(cumulative_optimal_reward - cumulative_reward))
    # print("average_reward_in_each_round", average_reward_in_each_round)
    axs[1].plot(X, average_reward_in_each_round, color='black', label='average reward')

    axs[1].set(xlabel='round number', ylabel='Average Reward per round')
    axs[1].grid(True)
    axs[1].legend(loc='lower right')
    axs[1].set_xlim(0, T)
    axs[1].set_ylim(0, 1.1*max(average_reward_in_each_round))
    # plt.savefig("./figures/prog3_figure.png")
    plt.show()


def plot_graph(T, cand_prices, average_reward_in_each_round, cum_rewards, average_arm_selections,
               optimal_revenue, max_cum_reward):
    """Plot a 3-row k-column graph with rolling average, cumulative reward and arm selection scatter plot."""
    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.plot(average_reward_in_each_round, label='Average rewards', color='palevioletred', alpha=0.8)
    plt.xlabel('Time-step t', fontsize=12)
    plt.ylabel(f'Avg. reward for classicBFG', fontsize=12)
    plt.title(f"classicBFG\nRolling Average Reward", fontsize=13)
    plt.axis([0, T, 0, optimal_revenue + 0.1])

    plt.subplot(3, 1, 2)
    plt.plot(cum_rewards, label='Cumulative reward', color='orange', alpha=0.8)
    plt.xlabel('Time-step t', fontsize=12)
    plt.ylabel(f'Cum. reward for classicBFG', fontsize=12)
    plt.title(f"Cumulative Reward", fontsize=13)
    plt.axis([0, T, 0, max_cum_reward])

    plt.subplot(3, 1, 3)
    # print(">>>> X value", range(T - 1))
    cm = plt.cm.get_cmap('YlOrBr')
    for n in range(len(cand_prices)):
        plt.scatter(range(T + 1), [n + 1]*(T + 1), s=2.5, c=average_arm_selections[n], alpha=.3, cmap=cm)

    plt.axis([0, T, 0, len(cand_prices) + 1])
    plt.xlabel('Time-step t', fontsize=12)
    plt.ylabel(f'classicBFG\'s arm selection', fontsize=12)
    plt.title("Arm selection scatter plot", fontsize=13)

    plt.tight_layout(pad=2.0)
    # plt.savefig('./figures/graph_demo.png')
    plt.show()


def generateDm(N, C):
    dm = np.zeros(shape=(N, C), dtype=int)
    for i in range(N):
        for j in range(C):
            dm[i][j] = int(random.gauss(100 + i, 0.1*i))
    return dm


def generateRm(dm, cand_prices):
    rm = np.zeros(shape=(len(dm), len(dm[0])), dtype=int)
    for i in range(len(dm)):
        for j in range(len(dm[0])):
            rm[i][j] = dm[i][j]

    # temp HACK
    for row in rm:
        for i in range(len(row)):
            row[i] *= cand_prices[i]
    return rm


def initialize_optimal_prices(N, rm, cand_prices):
    optimal_prices = []
    for i in range(N):
        idx = np.argmax(rm[i])
        optimal_prices.append(cand_prices[idx])
    return optimal_prices


def observeDemand(curr_price) -> int:
    """TODO: Preferably we can observe demand that's associated with a distribution, and 
             we should only return one value at a time, not the whole array"""
    # observedDemandArray = [0, 112, 64, 88, 74, 80]  # hard-coded (T is 1-based)
    # observedDemandArray = [random.randint(70, 140) for i in range(T+1)]
    return int(random.gauss(100 - 5*curr_price, 3))


def argmax_j(competitor_row):
    return np.argmax(competitor_row)


def argmin_i(rm_column):
    return np.argmin(rm_column)


def get_cum_revenue(r):
    cumulative_revenue = 0
    for k in r.keys():
        cumulative_revenue += sum(r[k])
    return cumulative_revenue


def main():
    # keys are the candidate pricings, and values are the calculated revenues associated with those key prices.
    r = dict()

    # N competitors: e.g. 3
    # C candidate prices: e.g. 4
    N = 3
    C = 4

    T = 10  # rounds (5000)
    iterations = 5  # total iteration
    avg_rewards = np.zeros(shape=(T + 1), dtype=float)
    cum_rewards = np.zeros(shape=(T + 1), dtype=float)
    average_cum_rewards = np.zeros(shape=(T + 1), dtype=float)
    cum_rewards_stacked = np.zeros(shape=(T + 1), dtype=float)
    running_optimal_revenue = 0
    # will be use to get the average_reward_in_each_round
    timestep_reward_stacked = np.zeros((T + 1), dtype=int)
    arm_selection_count = np.zeros(shape=(T + 1, C), dtype=float)
    average_arm_selections = np.zeros(shape=(C, T + 1), dtype=float)

    for iteration in range(iterations):
        print("#", iteration)
        # From competitor 1 to 3, we have decreasing avg. demand, but increasing variance between different prices
        # dm = [[230, 220, 210, 200], [100, 85, 70, 55], [80, 60, 40, 20]]
        dm = generateDm(N, C)
        cand_prices = [5,6,7,8]  # a.k.a. "p" in our original pseudocode
        rm = generateRm(dm, cand_prices)

        optimal_prices = initialize_optimal_prices(N, rm, cand_prices)

        # initialize i and curr_price for our first timestep
        i = random.choice(range(N))
        curr_price = optimal_prices[i]

        cand_num = cand_prices.index(curr_price)
        x = np.zeros(T + 1, dtype=int)  # array of observed demands

        for t in range(1, T + 1):
            # print(i, "\t", curr_price, "\t", cand_num)
            # curr_optimal_revenue = np.amax(rm)  # find maximum in 2D matrix
            # running_optimal_revenue = max(running_optimal_revenue, curr_optimal_revenue)
            x[t] = observeDemand(curr_price)
            reward = curr_price*x[t] # current revenue
            # print(curr_price,"*",x[t],"=",reward)
            running_optimal_revenue = max(running_optimal_revenue, reward)

            if curr_price not in r.keys():
                r[curr_price] = [curr_price*x[t]]
            else:
                r[curr_price].append(curr_price*x[t])

            avg_r_curr_price = sum(r[curr_price])/len(r[curr_price])

            # Update dm, rm, optimal price according to latest observed demand and revenue
            cand_num = cand_prices.index(curr_price)

            # THIS SECTION IS UPDATING VARIABLES FOR PLOTTING
            # don't need to explicitly persist r (revenue dictionary) because revenues are appended
            arm_selection_count[t][cand_num] += 1
            timestep_reward_stacked[t] += reward
            new_avg = (cum_rewards[t - 1] + reward)/(t + 1)
            avg_rewards[t] = new_avg
            # double_loop.py appends to cum_rewards in another way! IS THIS CORRECT TOO?
            # cum_rewards.append(new_avg*t)
            cum_rewards[t] = cum_rewards[t - 1] + reward

            dm[i][cand_num] = (dm[i][cand_num] + x[t])//2 # TODO: Try a weighted average
            rm[i][cand_num] = dm[i][cand_num]*curr_price
            optimal_prices[i] = cand_prices[argmax_j(rm[i])]

            # find the competitor with least difference in revenue
            i = argmin_i([np.abs(rm[j][cand_num] - avg_r_curr_price) for j in range(N)])
            print(i)
            curr_price = optimal_prices[i]

            # EXTRA DEBUG
            for j in range(N):
                print(np.abs(rm[j][cand_num]- avg_r_curr_price))
            print("-----------------------------------------")

        cur_max_r = 0
        for row in rm:
            for entry in row:
                cur_max_r = max(cur_max_r, entry)
        print("Final optimal revenue:", cur_max_r)

        # calculate cumulative revenue
        # print("Cumulative revenue: $", get_cum_revenue(r))

    average_reward_in_each_round = np.zeros(T + 1, dtype=float)
    # squash 5X100 -> 1X100
    for t in range(1, T + 1):
        average_reward_in_each_round[t] = float(timestep_reward_stacked[t])/float(iterations)

    for t in range(1, T + 1):
        for arm_index in range(C):
            average_arm_selections[arm_index][t] = float(arm_selection_count[t][arm_index])/float(iterations)

    cumulative_optimal_reward = 0.0
    cumulative_reward = 0.0
    x_axis = np.zeros(T + 1, dtype=int)
    regrets = np.zeros(T + 1, dtype=float)  # regret for each round

    for t in range(1, T + 1):
        x_axis[t] = t
        cumulative_optimal_reward += running_optimal_revenue
        cumulative_reward += average_reward_in_each_round[t]
        # print("cumulative_optimal_reward at t", t, "is", cumulative_optimal_reward)
        # print("cumulative_reward at t", t, "is", cumulative_reward)
        # print("average_reward_in_each_round at t", t, "is", average_reward_in_each_round[t])
        # print()``
        regrets[t] = max(0, cumulative_optimal_reward - cumulative_reward)
        # regrets[t] = cumulative_optimal_reward - cumulative_reward

    # print("x-axis", x_axis)
    # print("y-axis", regrets)
    # print("optimal revenue", optimal_revenue)
    # print("timestep_reward_stacked", timestep_reward_stacked)
    # print("Arm_selection_count", arm_selection_count)
    # print("Average_arm_selections", average_arm_selections)
    plot_regret(x_axis, regrets, cumulative_optimal_reward, cumulative_reward, average_reward_in_each_round, T)
    max_cum_reward = max(cum_rewards)
    plot_graph(T, cand_prices, average_reward_in_each_round, cum_rewards, average_arm_selections,
               running_optimal_revenue, max_cum_reward)

    # x_axis = np.zeros(shape=(T + 1), dtype=int)
    # regrets = np.zeros(shape=(T + 1), dtype=int)
    # cum_optimal_revenue = 0
    # cum_avg_revenue = 0
    # for t in range(1, T + 1):
    #     x_axis[t] = t
    #     cum_optimal_revenue += optimal_revenue
    #     cum_avg_revenue += average_reward_in_each_round[t]
    #     regrets[t] = max(0, cum_optimal_revenue - cum_avg_revenue)
    #     print("cum optimal revenue", cum_optimal_revenue)
    #     print("cum average revenue", cum_avg_revenue)
    # print(average_reward_in_each_round)
    # print(regrets)
    # plot_regret(x_axis, regrets, cum_optimal_revenue, cum_avg_revenue, average_reward_in_each_round, T + 1,
    #             "new_algorithm")


if __name__ == "__main__":
    main()
