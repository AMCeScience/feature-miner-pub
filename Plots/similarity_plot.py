import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import Libs.outcome_fetcher as fetcher
import Libs.similarity_fetcher as sim_fetch
import pickle, config, sys, pandas as pd, numpy as np


def plot():
  font = {'size': 16}

  matplotlib.rc('font', **font)

  plot_curves()


def plot_curves():
  # Get the cosine similarity matrix
  data = sim_fetch.get_similarity()

  # Get all the review names
  reviews = data.columns

  num_cols = len(reviews) - 1

  mean_arr = [0] * num_cols

  x = np.arange(1, 50, 1)

  plt.figure()

  # Loop over each review
  for review_name in reviews:
    # Get a review's cosine similarity row
    similarities = data.loc[[review_name]]

    # Remove the 1.0 similarity for the review itself (i.e. matrix diagonal)
    similarities = similarities.drop(review_name, axis = 1)
    # Sort the remaining values from highest to lowest
    similarities = similarities.sort_values(by = review_name, ascending = False, axis = 1)

    plt.plot(x, similarities.values.tolist()[0], alpha = 0.3)

    # Loop over the columns
    for i in range(0, len(similarities.columns)):
      item = similarities.iloc[:,i].item()

      # Calculate the mean for the column
      mean_arr[i] = mean_arr[i] + item

  mean_arr = [i / len(data.columns) for i in mean_arr]

  plt.plot(x, mean_arr, color = 'black', linestyle = 'dashed', label = 'Mean similarity')

  plt.xticks((1, 10, 20, 30, 40, 49), (1, 10, 20, 30, 40, 49))
  plt.xlim(1, 49)
  plt.xlabel('Review Rank (sorted highest to lowest)')

  plt.ylabel('Cosine Similarity')

  plt.legend()
  plt.tight_layout()
  plt.savefig(config.PLOT_LOCATION + '/cosine_similarity.pdf')
