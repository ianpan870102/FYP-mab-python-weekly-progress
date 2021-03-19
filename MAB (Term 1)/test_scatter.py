import matplotlib.pyplot as plt

def main():
  # for n in range(len(cand_prices)):
  cm = plt.cm.get_cmap('YlOrBr')
  # plt.subplot(3, 1, 3)
  plt.figure(figsize=(15, 10))
  plt.scatter([1, 2, 3, 4, 5], [10, 11, 12, 13, 14], s=2.5, c=[500, 10000, 3, 40000, 5], alpha=.3, cmap=cm)
  plt.axis([0, 6, 0, 16])
  plt.xlabel('Time-step t', fontsize=12)
  plt.ylabel(f'classicBFG\'s arm selection', fontsize=12)
  plt.title("Arm selection scatter plot", fontsize=13)
  plt.show()

main()