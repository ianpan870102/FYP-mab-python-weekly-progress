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
cand_prices = [5, 6, 7, 8]
T = 5
total_iteration = 3
demand_list = [[110, 105, 100, 95], [120, 89, 85, 81], [100, 90, 60, 50]]
# competitors and prices are 0-index based
N = len(demand_list)
C = len(demand_list[0])
optimal_revenue = 700

demand_matrix = np.zeros(shape=(N, C, T + 1), dtype=int)
avg_dem_matrix = np.zeros(shape=(N, C), dtype=int)

revenue_matrix = np.zeros(shape=(N, C, T + 1), dtype=int)
avg_rev_matrix = np.zeros(shape=(N, C), dtype=int)

# the first element is 0 because time is 1-index based in MAB for-loop
observed_demand = [0, 112, 64, 88, 74, 80]


def update_optimal_prices():
    opt_prices = []
    for i in range(N):
        idx = np.argmax(avg_rev_matrix[i])
        opt_prices.append(cand_prices[idx])
    return opt_prices


def update_avg_demand_revenue():
    for i in range(N):
        for j in range(C):
            avg_dem_matrix[i][j] = np.sum(demand_matrix[i][j])/np.count_nonzero(demand_matrix[i][j])
            avg_rev_matrix[i][j] = np.sum(revenue_matrix[i][j])/np.count_nonzero(revenue_matrix[i][j])


def main():
    cum_revenue_stacked = np.zeros(shape=(T + 1))
    '''MAB Phase'''
    rev_diff_matrix = np.zeros(shape=(N), dtype=int)

    # T is 1-index based
    for i in range(total_iteration):
        # fill revenue and demand matrix
        for i in range(N):
            for j in range(C):
                # This 3rd index is one as T is 1-based
                demand_matrix[i][j][1] = demand_list[i][j]
                # TODO: imporve fetching of price to find revenue
                revenue_matrix[i][j][1] = demand_matrix[i][j][1]*cand_prices[j]

        # Should this array be dynamic?
        x = np.zeros(T + 1, dtype=int)
        curr_price = random.choice(cand_prices)
        # curr_price = cand_prices[0]
        for t in range(1, T + 1):
            print("------------------------------------->")
            print("T:", t)
            print("Revenue matrix:\n", revenue_matrix)
            x[t] = observed_demand[t]
            # get index of curr_price
            cand_num = cand_prices.index(curr_price)
            print("Current price:", curr_price)
            print("Candidate price num:", cand_num + 1)
            curr_revenue = curr_price*x[t]
            print("Current revenue:", curr_revenue)
            cum_revenue_stacked[t] += curr_revenue  # persists over iterations

            # Update previously choose entry's demand and revenue matrix cell
            if t > 1:
                # append observed demand
                demand_matrix[comp_i][cand_num][t] = observed_demand[t]
                # append observed revenue
                revenue_matrix[comp_i][cand_num][t] = curr_revenue

            # re-calculate avg_rev_matrix and avg_dem_matrix
            update_avg_demand_revenue()

            print("Average revenue matrix:-----\n", avg_rev_matrix)

            # update optimal price after updating demand and revenue matrices
            opt_prices_list = update_optimal_prices()
            print("Optimal price list", opt_prices_list)
            '''Choosing revenue value to use in t+1'''
            # calculate revenue difference for chosen cand_num
            for i in range(N):
                rev_diff_matrix[i] = abs(avg_rev_matrix[i][cand_num] - curr_revenue)
            # min_diff = np.amin(rev_diff_matrix)
            print("Rev_diff_matrix:\n", rev_diff_matrix)
            # if multiple values are similar, the first one will be chosen
            comp_i = np.argmin(rev_diff_matrix)
            print("Competitor for next round:", comp_i + 1)
            # set curr_price to optimal price of competitor i
            curr_price = opt_prices_list[comp_i]
            print("Price for next round:", curr_price)
        # print("Cumulative revenue:", cum_revenue)

    average_reward_in_each_round = np.zeros(T + 1, dtype=float)
    for t in range(1, T + 1):
        average_reward_in_each_round[t] = cum_revenue_stacked[t]/total_iteration

    x_axis = np.zeros(shape=(T + 1), dtype=int)
    regrets = np.zeros(shape=(T + 1), dtype=int)
    cum_optimal_revenue = 0
    cum_avg_revenue = 0
    for t in range(1, T + 1):
        x_axis[t] = t
        cum_optimal_revenue += optimal_revenue
        cum_avg_revenue += average_reward_in_each_round[t]
        regrets[t] = cum_optimal_revenue - cum_avg_revenue
        print("cum optimal revenuce", cum_optimal_revenue)
        print("cum average revenuce", cum_avg_revenue)
    print(average_reward_in_each_round)
    print(regrets)
    plot_regret(x_axis, regrets, cum_optimal_revenue, cum_avg_revenue, average_reward_in_each_round, T + 1,
                "new_algorithm")


if __name__ == "__main__":
    main()
