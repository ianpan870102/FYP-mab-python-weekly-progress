import random
from arms.bernoulli import BernoulliArm
from algorithms.epsilon_greedy.standard_epsilon import EpsilonGreedy
from testing_framework.tests import test_algorithm


def argmax(arr):
  return arr.index(max(arr))


random.seed(1)
means = [0.1, 0.1, 0.1, 0.1, 0.9]
random.shuffle(means)
arms = list(map(lambda mu: BernoulliArm(mu), means))
print("Best arm is " + str(argmax(means)))

with open("algorithms/epsilon_greedy/standard_epsilon_results.tsv", "w") as f:
  for epsilon in [0.1, 0.2, 0.3, 0.4, 0.5]:
    algo = EpsilonGreedy(epsilon, [], [])
    algo.initialize(len(means))
    results = test_algorithm(algo, arms, 5000, 250)
    for i in range(len(results[0])):
      f.write(str(epsilon) + "\t")
      f.write("\t".join([str(results[j][i]) for j in range(len(results))]) + "\n")
