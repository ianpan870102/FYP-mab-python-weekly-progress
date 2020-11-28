import random
import numpy as np
import random
from arms.bernoulli import BernoulliArm
from arms.normal import NormalArm
from algorithms.new_algorithm.new_algorithm import new_algorithm

# Initialisation and Preprocess phase
cand_prices = [5, 6, 7, 8]
T = 5
demand_list = [[110, 105, 100, 95], [120, 89, 85, 81], [100, 90, 60, 50]]
# competitors, prices are 0-index based
N = len(demand_list)
C = len(demand_list[0])
demand_matrix = np.zeros(shape=(N, C, T), dtype=int)

# The 3rd parameter sohuldn't be static
revenue_matrix = np.zeros(shape=(N, C, T), dtype=int)

# fill revenue and demand matrix
for i in range(N):
    for j in range(C):
        demand_matrix[i][j][0] = demand_list[i][j]
        # imporve fetching of price to find revenue
        revenue_matrix[i][j][0] = demand_matrix[i][j][0]*cand_prices[j]

print("Demand matrix is\n", demand_matrix)
x = np.zeros(T, dtype=int)
# This is not a good way of recording observer_demand (must be dynamic)
# the first element is 0 because time is 1-index based in MAB for-loop
observed_demand = [0, 112, 64, 88, 74, 80]
avg_rev_matrix = np.zeros(shape=(N, C), dtype=int)


def updateOptimalPrices():
    opt_prices = []
    for i in range(N):
        for j in range(C):
            avg_rev_matrix[i][j] = np.sum(revenue_matrix[i][j])/np.count_nonzero(revenue_matrix[i][j])
    for i in range(N):
        idx = np.argmax(avg_rev_matrix[i])
        opt_prices.append(cand_prices[idx])
    return opt_prices


def main():
    # curr_price = random.choice(cand_prices)
    curr_price = cand_prices[0]
    print("Revenue matrix\n", revenue_matrix)
    print("------------------------------------->")
    # MAB phase
    prev_i = -1
    rev_diff_matrix = np.zeros(shape=(N, C), dtype=int)

    # T is 1-index based
    for t in range(1, T + 1):
        x[t] = observed_demand[t]
        curr_revenue = curr_price*x[t]
        print("Curr revernue", curr_revenue)
        print("About to print avg_rev_matrix")
        # update optimal price after updating demand and revenue matrices?
        opt_prices_list = updateOptimalPrices()
        if t == 1:
            # calculate revenue difference
            for i in range(N):
                for j in range(C):
                    # improve fetching of price to find revenue
                    # print(revenue_matrix[i][j][0])
                    # avg_revenue = np.sum(revenue_matrix[i][j])/revenue_matrix[i][j].shape
                    # print(avg_revenue)
                    rev_diff_matrix[i][j] = revenue_matrix[i][j][0] - curr_revenue
                    # print("Price difference", i, j, rev_diff_matrix[i][j])
            # min_diff = np.amin(rev_diff_matrix)
            min_index = np.argmin(rev_diff_matrix)
            # Update the average revenue
            # This is an array of existing revenues

            # index of revenue_matrix to upload
            # we have to use shape of rev_diff_matrix because we use that matrix's min index
            upd_index = np.unravel_index(min_index, rev_diff_matrix.shape)
            comp_i = upd_index[0]
            prev_i = comp_i
            # set curr_price to optimal price of competitor i
            curr_price = opt_prices_list[comp_i]
            cand_num = curr_price
        else:
            print("This is before updating array", revenue_matrix[upd_index])
            # append observed revenue
            revenue_matrix[upd_index][t] = curr_revenue

            print("This is after updating array", revenue_matrix[upd_index])
            # not sure why we need in pseudocode
            avg_r_curr_price = np.sum(revenue_matrix[upd_index])/np.count_nonzero(revenue_matrix[upd_index])

            # Update dm, rm, optimal price according to lastest observed demand & revenue
            # which competitor we chose
            prev_i = opt_prices_list.index(curr_price)
            # which price we chose
            cand_num = cand_prices.index(curr_price)

            demand_matrix[prev_i][cand_num] = (demand_matrix[prev_i][cand_num] + x[t])/2
            revenue_matrix[prev_i][cand_num] = demand_matrix[prev_i][cand_num]*curr_price
            opt_prices_list = updateOptimalPrices()

            # chosen_competitor =

            # print(min_diff)
            # print(min_index)
            # print(rev_diff_matrix)


if __name__ == "__main__":
    main()
