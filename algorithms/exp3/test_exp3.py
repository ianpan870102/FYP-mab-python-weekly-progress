import random
from arms.bernoulli import BernoulliArm
from algorithms.exp3.exp3 import Exp3
from testing_framework.tests import test_algorithm


def argmax(arr):
  return arr.index(max(arr))


random.seed(1)
means = [0.1, 0.1, 0.1, 0.1, 0.9]
n_arms = len(means)
random.shuffle(means)
arms = map(lambda mu: BernoulliArm(mu), means)
print("Best arm is " + str(argmax(means)))

with open("algorithms/exp3/exp3_results.tsv", "w") as f:
  for exp3_gamma in [0.1, 0.2, 0.3, 0.4, 0.5]:
    algo = Exp3(exp3_gamma, [])
    algo.initialize(n_arms)
    results = test_algorithm(algo, arms, 5000, 250)
    for i in range(len(results[0])):
      f.write(str(exp3_gamma) + "\t")
      f.write("\t".join([str(results[j][i]) for j in range(len(results))]) + "\n")
