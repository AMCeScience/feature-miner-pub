import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import Libs.outcome_fetcher as fetcher
import config, numpy as np

def plot():
  font = {'size': 16}

  matplotlib.rc('font', **font)

  outcome_fetcher = fetcher.Outcome_fetcher()
  matrix = outcome_fetcher.get_data('onevsone')

  plot_ordered_heatmap(matrix)
  plot_unordered_heatmap(matrix)
  plot_side_by_side(matrix)


def set_diagonal(matrix, value = 1):
  # Make a copy of the object as numpy works in-place
  copy = matrix.copy()
  np.fill_diagonal(copy, value)

  return copy


def order_matrix(matrix):
  # Order by mean row value
  matrix = matrix[np.mean(matrix, axis = 1).argsort()]
  # Order by mean column value
  matrix = matrix[:,np.mean(matrix, axis = 0).argsort()]

  return matrix


def plot_ordered_heatmap(matrix, new_figure = True):
  # Remove the diagonal to remove the influence
  # of self testing from the heatmap
  matrix = set_diagonal(matrix, 0)
  # Order the matrix by mean row and column values
  matrix = order_matrix(matrix)

  # Reset matplotlib if necessary
  if new_figure:
    fig = plt.figure()

  # Plot heatmap
  plt.imshow(matrix, cmap = 'hot', interpolation = 'nearest')

  plt.xlabel('Test Review Index\n(ordered by mean column value)')
  plt.tick_params(axis = 'x', bottom = False, labelbottom = False)
  plt.ylabel('Training Review Index\n(ordered by mean row value)')
  plt.tick_params(axis = 'y', left = False, labelleft = False)

  plt.tight_layout()

  # Reset matplotlib if necessary
  if new_figure:
    plt.savefig(config.PLOT_LOCATION + '/heatmap.pdf')
    plt.close(fig)


def plot_unordered_heatmap(matrix, new_figure = True):
  # Reset matplotlib if necessary
  if new_figure:
    fig = plt.figure()

  # Plot heatmap
  plt.imshow(matrix, cmap = 'hot', interpolation = 'nearest')

  plt.xlabel('Test Review Index')
  plt.ylabel('Training Review Index')

  plt.tight_layout()

  # Reset matplotlib if necessary
  if new_figure:
    plt.savefig(config.PLOT_LOCATION + '/unordered_heatmap.pdf')
    plt.close(fig)


def plot_side_by_side(matrix):
  font = {'size': 12}

  matplotlib.rc('font', **font)

  plt.subplot(1, 2, 1)
  plt.title('(a)')
  plot_unordered_heatmap(matrix, False)

  plt.subplot(1, 2, 2)
  plt.title('(b)')
  plot_ordered_heatmap(matrix, False)

  plt.tight_layout()
  plt.savefig(config.PLOT_LOCATION + '/double_heatmap.pdf')