import random
from arms.bernoulli import BernoulliArm
from algorithms.ucb.ucb1 import UCB1
from testing_framework.tests import test_algorithm


def argmax(arr):
  return arr.index(max(arr))


random.seed(1)
means = [0.1, 0.1, 0.1, 0.1, 0.9]
n_arms = len(means)
random.shuffle(means)
arms = map(lambda mu: BernoulliArm(mu), means)
print("Best arm is " + str(argmax(means)))

algo = UCB1([], [])
algo.initialize(n_arms)
results = test_algorithm(algo, arms, 5000, 250)

with open("algorithms/ucb/ucb1_results.tsv", "w") as f:
  for i in range(len(results[0])):
    f.write("\t".join([str(results[j][i]) for j in range(len(results))]) + "\n")
