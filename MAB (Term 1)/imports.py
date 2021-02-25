import matplotlib.pyplot as plt
import random
from arms.adversarial import AdversarialArm
from arms.bernoulli import BernoulliArm
from arms.normal import NormalArm
from algorithms.epsilon_greedy.standard_epsilon import EpsilonGreedy
from algorithms.epsilon_greedy.annealing_epsilon import AnnealingEpsilonGreedy
from algorithms.softmax.standard_softmax import Softmax
from algorithms.softmax.annealing_softmax import AnnealingSoftmax
from algorithms.ucb.ucb1 import UCB1
from algorithms.ucb.ucb2 import UCB2
from algorithms.exp3.exp3 import Exp3
from algorithms.hedge.hedge import Hedge


def argmax(arr):
  return arr.index(max(arr))
