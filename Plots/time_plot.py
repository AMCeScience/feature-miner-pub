import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import Libs.outcome_fetcher as fetcher
from scipy import stats

def plot():
  font = {'size': 16}

  matplotlib.rc('font', **font)

  outcome_fetcher = fetcher.Outcome_fetcher()
  full_data = outcome_fetcher.get_timing_data('leaveoneout')
  timing_data = outcome_fetcher.get_timing_data('nvsone')

  flat_full_data = full_data[0]
  flat_timing_data = [i for sub in timing_data[0] for i in sub]

  plt.figure(figsize=(5,6))
  plt.boxplot([flat_timing_data, flat_full_data], widths = (0.4, 0.4))

  plt.xticks((1, 2), ('1', '49'))
  plt.xlabel('Number of reviews in training set')
  plt.ylim(0, 600)
  plt.ylabel('Training duration (seconds)')

  plt.tight_layout()
  plt.savefig(config.PLOT_LOCATION + '/training_duration.pdf')

  print(stats.ranksums(flat_timing_data, flat_full_data).pvalue)