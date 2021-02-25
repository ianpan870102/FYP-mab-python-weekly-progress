import random
import numpy as np
import random
import matplotlib.pyplot as plt
from arms.bernoulli import BernoulliArm
from arms.normal import NormalArm

from matplotlib import rcParams
rcParams['font.family'] = ['Roboto']
for w in ["font.weight", "axes.labelweight", "axes.titleweight", "figure.titleweight"]:
    rcParams[w] = 'regular'


def plot_regret(X, Y, cumulative_optimal_reward, cumulative_reward, average_reward_in_each_round, T, algo_name):
    fig, axs = plt.subplots(2)  # get two figures, top is regret, bottom is average reward in each round
    fig.suptitle(f'Performance of {algo_name}')
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
    plt.savefig("./figures/prog3_figure.png")
    plt.show()


'''Initialisation and Preprocess phase'''
# C candidate prices, N competitors (demand functions)
cand_prices = [5, 6, 7, 8] # a.k.a. "p"
T = 5  # rounds
total_iteration = 3

# 3 competitors, 4 candidate prices
dm = [[110, 105, 100, 95], [120, 89, 85, 81], [100, 90, 60, 50]]
rm = generateRm()
optimal_prices = initialize_optimal_prices()
# competitors and prices are 0-index based
N = len(dm)
C = len(dm[0])  # len(cand_prices)
optimal_revenue = 700  # TODO: hard-coded

# the first element is 0 because time is 1-index based in MAB for-loop


def generateRm():
    rm = dm
    for row in rm:
        for i in range(len(row)):
            row[i] *= cand_prices[i]
    return rm


def observeDemand():
    """TODO: Preferably we can observe demand that's associated with a distribution, and 
             we should only return one value at a time, not the whole array"""
    observed_demand = [0, 112, 64, 88, 74, 80]  # hard-coded (T is 1-based)
    return observed_demand


# def update_optimal_prices():
#     opt_prices = []
#     for i in range(N):
#         idx = np.argmax(rm[i])
#         opt_prices.append(cand_prices[idx])
#     return opt_prices

# def update_avg_demand_revenue():
#     for i in range(N):
#         for j in range(C):
#             dm[i][j] = np.sum(demand_matrix[i][j])/np.count_nonzero(demand_matrix[i][j])
#             rm[i][j] = np.sum(revenue_matrix[i][j])/np.count_nonzero(revenue_matrix[i][j])


def initialize_optimal_prices():
    optimal_prices = []
    for i in range(N):
        idx = np.argmax(rm[i])
        optimal_prices.append(cand_prices[idx])
    return optimal_prices


def main():
    curr_price = random.choice(cand_prices)
    x = np.zeros(T + 1, dtype=int)  # array of observed demands

    # keys are the candidate pricings, and values are the calculated revenues associated with those key prices.
    r = dict()

    for t in range(1, T + 1):
        x[t] = observeDemand()[t]

        if curr_price not in r.keys():
            r[curr_price] = [curr_price*x[t]]
        else:
            r[curr_price].append(curr_price*x[t])

        avg_r_curr_price = sum(r[curr_price])/len(r[curr_price])

        # Update dm, rm, optimal price according to latest observed demand and revenue
        # prev_i is the index of the competitor whose optimal price we chose
        prev_i = optimal_prices.index(curr_price)
        cand_num = p.index(curr_price)
        dm[prev_i][cand_num ] = (dm[prev_i][cand_num ] + x[t]) / 2
        rm[prev_i][cand_num] = dm[prev_i][cand_num] * curr_price
        optimal_prices[prev_i] = cand_prices[argmax_j(rm[i][j])] # TODO argmax_j() requires implementation







        if t > 1:
            # append observed demand
            demand_matrix[comp_i][cand_num][t] = observed_demand[t]
            # append observed revenue
            revenue_matrix[comp_i][cand_num][t] = curr_revenue

        # re-calculate rm and dm
        update_avg_demand_revenue()

        # update optimal price after updating demand and revenue matrices
        opt_prices_list = update_optimal_prices()
        # calculate revenue difference for chosen cand_num
        for i in range(N):
            rev_diff_matrix[i] = abs(rm[i][cand_num] - curr_revenue)
        # min_diff = np.amin(rev_diff_matrix)
        # if multiple values are similar, the first one will be chosen
        # set curr_price to optimal price of competitor i
        curr_price = opt_prices_list[comp_i]

    x_axis = np.zeros(shape=(T + 1), dtype=int)
    regrets = np.zeros(shape=(T + 1), dtype=int)
    cum_optimal_revenue = 0
    cum_avg_revenue = 0
    for t in range(1, T + 1):
        x_axis[t] = t
        cum_optimal_revenue += optimal_revenue
        cum_avg_revenue += average_reward_in_each_round[t]
        regrets[t] = max(0, cum_optimal_revenue - cum_avg_revenue)
        print("cum optimal revenue", cum_optimal_revenue)
        print("cum average revenue", cum_avg_revenue)
    print(average_reward_in_each_round)
    print(regrets)
    plot_regret(x_axis, regrets, cum_optimal_revenue, cum_avg_revenue, average_reward_in_each_round, T + 1,
                "new_algorithm")


if __name__ == "__main__":
    main()