import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import Libs.outcome_fetcher as fetcher
import Database.db_connector as db
import config, numpy as np
from scipy import stats
import statsmodels.stats.multitest as multi
import pandas as pd
import os

def plot():
  font = {'size': 16}

  matplotlib.rc('font', **font)

  outcome_fetcher = fetcher.Outcome_fetcher()
  full_data = outcome_fetcher.get_data('leave_one_out')

  if not os.path.exists(config.PLOT_LOCATION):
    os.makedirs(os.path.dirname(config.PLOT_LOCATION))

  section_feature(full_data)
  section_classifier_and_feature(full_data)
  section_classifier(full_data)
  section_reviews(full_data)


# Plots a boxplot per feature type
def section_feature(data):
  def sect(x):
    # Get the first cell of a apply_along_axis value.
    first_cell = x.tolist()[0][0]

    # Check whether the cell contains the string in 'feature'.
    return feature in first_cell

  # Retrieve all rows with TF outcomes.
  feature = 'TF'
  tf_rows = np.apply_along_axis(sect, axis = 1, arr = data)
  # Retrieve all rows with TM outcomes.
  feature = 'TM'
  tm_rows = np.apply_along_axis(sect, axis = 1, arr = data)

  # Flatten matrices, turn into lists.
  tf_values = data[tf_rows,1:].ravel().astype(float).tolist()[0]
  tm_values = data[tm_rows,1:].ravel().astype(float).tolist()[0]

  #### SIGNIFICANCE

  significance = stats.ranksums(tf_values, tm_values).pvalue

  df = pd.DataFrame(significance, columns = ['TM (mean: %f)' % np.mean(tm_values)], index = ['TF (mean: %f)' % np.mean(tf_values)])

  filename = config.OUTPUT_LOCATION + '/significance_testing/feature_significance.xlsx'

  if not os.path.exists(filename):
    os.makedirs(os.path.dirname(filename))

  df.to_excel(filename, engine = 'openpyxl')

  #### PLOT

  y = [tf_values, tm_values]

  plt.figure()
  plt.boxplot(y)
  # Add a zero line
  plt.axhline(y = 0, alpha = 0.4, color = 'gray', linewidth = 1, linestyle = 'dashed')

  plt.xticks((1, 2), ('TF-IDF (N = %i)' % len(tf_values), 'TM (N = %i)' % len(tm_values)))
  plt.ylim(-0.2, 1)
  plt.ylabel('WSS@95')

  plt.tight_layout()

  filename = config.PLOT_LOCATION + '/preliminary_experiment/feature_plot.pdf'

  if not os.path.exists(filename):
    os.makedirs(os.path.dirname(filename))

  plt.savefig(filename)


def section_classifier_and_feature(data):
  def sect(x):
    first_cell = x.tolist()[0][0]

    return classifier in first_cell

  classifier = 'TM - rf'
  tm_rf_rows = np.apply_along_axis(sect, axis = 1, arr = data)
  classifier = 'TM - svm'
  tm_svm_rows = np.apply_along_axis(sect, axis = 1, arr = data)
  classifier = 'TF - rf'
  tf_rf_rows = np.apply_along_axis(sect, axis = 1, arr = data)
  classifier = 'TF - svm'
  tf_svm_rows = np.apply_along_axis(sect, axis = 1, arr = data)

  tm_rf_values = data[tm_rf_rows,1:].ravel().astype(float).tolist()[0]
  tm_svm_values = data[tm_svm_rows,1:].ravel().astype(float).tolist()[0]
  tf_rf_values = data[tf_rf_rows,1:].ravel().astype(float).tolist()[0]
  tf_svm_values = data[tf_svm_rows,1:].ravel().astype(float).tolist()[0]

  #### SIGNIFICANCE

  classifiers = ['TM - rf', 'TM - svm', 'TF - rf', 'TF - svm']

  arr = list()

  # Loop over every classifier
  for i in range(0, len(classifiers)):
    # Loop over every classifier
    for j in range(0, len(classifiers)):
      # Fetch the data for classifier i
      classifier = classifiers[i]
      a = np.apply_along_axis(sect, axis = 1, arr = data)
      a = data[a,1:].ravel().astype(float).tolist()[0]

      # Fetch the data for classifier j
      classifier = classifiers[j]
      b = np.apply_along_axis(sect, axis = 1, arr = data)
      b = data[b,1:].ravel().astype(float).tolist()[0]

      # Calculate Wilcoxon rank sum test for every classifier
      # against every other classifier
      p_val = stats.ranksums(a, b).pvalue

      arr.append(p_val)

  labels = classifiers.copy()

  for i in range(0, len(classifiers)):
    classifier = classifiers[i]
    a = np.apply_along_axis(sect, axis = 1, arr = data)
    a = data[a,1:].ravel().astype(float).tolist()[0]

    labels[i] = labels[i] + ' (mean = %f)' % np.mean(a)

  # Adjust p-values using Holm-Sidak method
  adjusted = multi.multipletests(arr, method = 'bonferroni')
  # Create list of lists with adjusted p-values,
  # one list per initial pool size
  restruct_arr = list(chunks(adjusted[1], len(classifiers)))

  df = pd.DataFrame(restruct_arr, columns = labels, index = labels)

  df.to_excel(config.OUTPUT_LOCATION + '/significance_testing/classifier_features_significance.xlsx', engine = 'openpyxl')

  #### PLOT

  y = [tm_rf_values, tm_svm_values, tf_rf_values, tf_svm_values]

  plt.figure()
  plt.boxplot(y)
  # Add a zero line
  plt.axhline(y = 0, alpha = 0.4, color = 'gray', linewidth = 1, linestyle = 'dashed')

  plt.xticks((1, 2, 3, 4), ('TM\nRF', 'TM\nSVM', 'TF-IDF\nRF', 'TF-IDF\nSVM'))
  plt.ylim(-0.2, 1)
  plt.ylabel('WSS@95')

  plt.tight_layout()
  plt.savefig(config.PLOT_LOCATION + '/preliminary_experiment/classifier_feature_plot.pdf')


# Plots a boxplot per classifier
def section_classifier(data):
  def sect(x):
    first_cell = x.tolist()[0][0]

    return classifier in first_cell

  classifier = 'rf'
  rf_rows = np.apply_along_axis(sect, axis = 1, arr = data)
  classifier = 'svm'
  svm_rows = np.apply_along_axis(sect, axis = 1, arr = data)

  rf_values = data[rf_rows,1:].ravel().astype(float).tolist()[0]
  svm_values = data[svm_rows,1:].ravel().astype(float).tolist()[0]

  #### SIGNIFICANCE

  significance = stats.ranksums(rf_values, svm_values).pvalue

  df = pd.DataFrame(significance, columns = ['RF (mean: %f)' % np.mean(rf_values)], index = ['SVM (mean: %f)' % np.mean(svm_values)])

  df.to_excel(config.OUTPUT_LOCATION + '/significance_testing/classifier_significance.xlsx', engine = 'openpyxl')

  #### PLOT

  y = [rf_values, svm_values]

  plt.figure()
  plt.boxplot(y)
  # Add a zero line
  plt.axhline(y = 0, alpha = 0.4, color = 'gray', linewidth = 1, linestyle = 'dashed')

  plt.xticks((1, 2), ('RF (N = %i)' % len(rf_values), 'SVM (N = %i)' % len(svm_values)))
  plt.ylim(-0.2, 1)
  plt.ylabel('WSS@95')

  plt.tight_layout()
  plt.savefig(config.PLOT_LOCATION + '/preliminary_experiment/classifier_plot.pdf')


# Plots a boxplot per review
def section_reviews(data):
  font = {'size': 12}

  matplotlib.rc('font', **font)

  # Cut first column as it contains an identifier.
  data = data[:,1:].astype(float)

  # Calculate and sort the medians per dataset.
  sorted_medians = np.median(data, axis = 0).argsort().tolist()[0]

  conn = db.Connector()

  # Fetch the review names, used for the xtick labels.
  review_names = conn.get_review_names()

  # Sort the review names.
  sorted_names = np.array(review_names)[sorted_medians]

  # Sort the data matrix.
  sorted_data = data[:,sorted_medians]

  # Split into rows.
  rows = np.split(sorted_data, sorted_data.shape[1], axis = 1)

  # Turn the splits into lists.
  rows = [row.tolist() for row in rows]
  # Flatten lists.
  rows = [[item for sublist in row for item in sublist] for row in rows]

  x_ticks = np.arange(1, len(sorted_names) + 1)

  plt.figure()

  # Add a zero line
  plt.axhline(y = 0, alpha = 0.4, color = 'gray', linewidth = 1, linestyle = 'dashed')

  # Add point cloud over boxplot
  for i in x_ticks:
    y_points = rows[i - 1]
    # Add jitter so that the points are not on
    # a straight line in the plot
    x = np.random.normal(i, 0.04, size = len(y_points))

    plt.plot(x, y_points, 'r.', alpha = 0.3)

  plt.xticks(x_ticks, (sorted_names), rotation = 90)
  plt.xlabel('Review')
  plt.ylim(-0.2, 1)
  plt.ylabel('WSS@95')

  # Adjust the left and right margins
  plt.gca().margins(x = 0.01)
  # Create a canvas
  plt.gcf().canvas.draw()
  # Get the widest tick label
  tl = plt.gca().get_xticklabels()
  print(tl)
  maxsize = max([t.get_window_extent().width for t in tl])

  # Calculate new width and adjust the plot
  added_margin = 1
  new_tick_width = maxsize / plt.gcf().dpi * len(x_ticks) + 2 * added_margin
  plt.gcf().set_size_inches(new_tick_width, plt.gcf().get_size_inches()[1])

  plt.tight_layout()
  plt.savefig(config.PLOT_LOCATION + '/review_plot.pdf')


def chunks(l, n):
  for i in range(0, len(l), n):
    yield l[i:i + n]
