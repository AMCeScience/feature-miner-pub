import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D

from collections import Counter
import Libs.file_storage as file_handle
import Libs.outcome_fetcher as fetcher
import config, numpy as np
import Libs.similarity_fetcher as sim_fetch
from scipy import stats
import statsmodels.stats.multitest as multi
import pandas as pd

class Nvsone_plot(object):

  colors = [0, 2, 1, 3, 4, 5, 6, 7, 8, 9]

  nvsone_matrix = None
  leaveoneout_matrix = None
  review_names = None
  similarity_matrix = None
  run_similarities = None

  def __init__(self):
    self.singles = config.GROUPS['singles']

    groups = {k: v for k, v in config.GROUPS.items() if k != 'singles'}
    self.groups = [x for v in groups.values() for x in v]

    self.plot()


  def plot(self):
    self.similarity_matrix = sim_fetch.get_similarity()
    np.fill_diagonal(self.similarity_matrix.values, 0)

    outcome_fetcher = fetcher.Outcome_fetcher()
    self.nvsone_matrix, self.leaveoneout_matrix, self.run_similarities, self.review_names = outcome_fetcher.get_data('n_vs_one')

    self.nvsone_random_matrix = outcome_fetcher.get_data('n_vs_one_random')

    self.group_compare()
    self.mean_compare()
    self.overlay_compare()
    self.wss_against_similarity()
    self.training_selection_distribution()
    self.mean_compare_including_random()


  @staticmethod
  def chunks(l, n):
    for i in range(0, len(l), n):
      yield l[i:i + n]


  def group_compare(self):
    font = {'size': 13}

    matplotlib.rc('font', **font)

    sorting_idx = np.median(self.leaveoneout_matrix, axis = 0).argsort().tolist()[0]

    nvsone_labels = config.SIMILARITY_STEPS

    clean_nvsone_matrix = self.nvsone_matrix[:, 1:]
    num_runs = int(self.nvsone_matrix.shape[0] / len(nvsone_labels))

    # Sort the review names
    sorted_names = np.array(self.review_names)[sorting_idx]
    sorted_names = sorted_names.tolist()

    # Sort the data matrices
    sorted_leaveoneout_data = self.leaveoneout_matrix[:, sorting_idx]
    sorted_nvsone_data = clean_nvsone_matrix[:, sorting_idx]

    single_idx = [sorted_names.index(x) for x in self.singles]
    group_idx = [sorted_names.index(x) for x in self.groups]

    single_leaveoneout_data = sorted_leaveoneout_data[:, single_idx].ravel().tolist()[0]
    group_leaveoneout_data = sorted_leaveoneout_data[:, group_idx].ravel().tolist()[0]

    single_nvsone_data = sorted_nvsone_data[:, single_idx]
    group_nvsone_data = sorted_nvsone_data[:, group_idx]

    def summarise(data):
      size_data = []

      # Loop over the similarity steps (i.e. training sizes)
      for i in range(1, len(nvsone_labels) + 1):
        nvsone_sim_step_values = []

        # Loop over all runs for this similarity step
        for run_num in range(1, num_runs + 1):
          # Concatenate the performance outcomes for all runs
          nvsone_sim_step_values = nvsone_sim_step_values + data[(i * run_num) - 1, :].tolist()

        size_data.append(nvsone_sim_step_values)

      return size_data

    def summarise_lines(data):
      size_data_low = []
      size_data_median = []
      size_data_high = []

      # Loop over the similarity steps (i.e. training sizes)
      for i in range(1, len(nvsone_labels) + 1):
        nvsone_sim_step_values = []

        # Loop over all runs for this similarity step
        for run_num in range(1, num_runs + 1):
          # Concatenate the performance outcomes for all runs
          nvsone_sim_step_values = nvsone_sim_step_values + data[(i * run_num) - 1, :].tolist()

        size_data_low.append(np.percentile(nvsone_sim_step_values, 25))
        size_data_median.append(np.median(nvsone_sim_step_values))
        size_data_high.append(np.percentile(nvsone_sim_step_values, 75))

      return size_data_low, size_data_median, size_data_high

    y_groups = summarise(group_nvsone_data)
    y_groups.append([group_leaveoneout_data])
    y_singles = summarise(single_nvsone_data)
    y_singles.append([single_leaveoneout_data])

    y_groups_low, y_groups_median, y_groups_high = summarise_lines(group_nvsone_data)
    y_groups_low.append(np.percentile(group_leaveoneout_data, 25))
    y_groups_median.append(np.median(group_leaveoneout_data))
    y_groups_high.append(np.percentile(group_leaveoneout_data, 75))

    y_singles_low, y_singles_median, y_singles_high = summarise_lines(single_nvsone_data)
    y_singles_low.append(np.percentile(single_leaveoneout_data, 25))
    y_singles_median.append(np.median(single_leaveoneout_data))
    y_singles_high.append(np.percentile(single_leaveoneout_data, 75))

    x_labels = [str(i) for i in config.SIMILARITY_STEPS] + ['49']

    #### PLOT

    x_range = range(0, len(x_labels))

    plt.figure()

    plt.plot(y_groups_median, marker = 's', label = '1 - 7 median', color = 'blue')
    plt.plot(y_groups_high, color = 'blue', linewidth = 0.2)
    plt.plot(y_groups_low, color = 'blue', linewidth = 0.2)
    plt.fill_between(x_range, y_groups_high, y_groups_low, alpha = 0.1, color = 'blue')

    plt.plot(y_singles_median, marker = 'o', label = 'Other median', color = 'red')
    plt.plot(y_singles_high, color = 'red', linewidth = 0.2)
    plt.plot(y_singles_low, color = 'red', linewidth = 0.2)
    plt.fill_between(x_range, y_singles_high, y_singles_low, alpha = 0.1, color = 'red')

    plt.xticks(x_range, x_labels)
    plt.xlabel('Number of reviews in training set')
    plt.ylim(0, 1)
    plt.ylabel('WSS@95')
    plt.legend()

    plt.tight_layout()
    plt.savefig(config.PLOT_LOCATION + '/single_group_split_compare.pdf')

    #### SEPARATE PLOT

    fig, (ax1, ax2) = plt.subplots(1, 2, sharey = True, figsize = (8,5))
    ax1.plot(y_groups_median, marker = 'o', label = 'Median', color = 'blue')
    ax1.plot(y_groups_high, color = 'blue', linewidth = 0.2)
    ax1.plot(y_groups_low, color = 'blue', linewidth = 0.2)
    ax1.fill_between(x_range, y_groups_high, y_groups_low, alpha = 0.1, color = 'blue', label = '$25^{th}$ to $75^{th}$ percentile')

    ax1.set_title('1 - 7')
    ax1.set_xticks(x_range)
    ax1.set_xticklabels(x_labels)

    ax2.plot(y_singles_median, marker = 'o', label = 'Median', color = 'red')
    ax2.plot(y_singles_high, color = 'red', linewidth = 0.2)
    ax2.plot(y_singles_low, color = 'red', linewidth = 0.2)
    ax2.fill_between(x_range, y_singles_high, y_singles_low, alpha = 0.1, color = 'red', label = '$25^{th}$ to $75^{th}$ percentile')

    ax2.set_title('Other')
    ax2.set_xticks(x_range)
    ax2.set_xticklabels(x_labels)

    ax1.set_ylim(0, 1)
    ax1.legend()
    ax2.set_ylim(0, 1)
    ax2.legend()

    fig.add_subplot(111, frameon = False)

    plt.tick_params(labelcolor = 'none', top = False, bottom = False, left = False, right = False)
    plt.grid(False)
    plt.xlabel('Number of reviews in training set')
    plt.ylabel('WSS@95')

    plt.tight_layout()
    plt.savefig(config.PLOT_LOCATION + '/single_group_split_compare_separate.pdf')

    font = {'size': 12}

    matplotlib.rc('font', **font)


  def mean_compare(self):
    font = {'size': 14}

    matplotlib.rc('font', **font)

    y_low = []
    y_median = []
    y_high = []

    nvsone_labels = config.SIMILARITY_STEPS
    num_runs = int(self.nvsone_matrix.shape[0] / len(nvsone_labels))

    # Loop over the similarity steps (i.e. training sizes)
    for i in range(1, len(nvsone_labels) + 1):
      nvsone_sim_step_values = []

      # Loop over all runs for this similarity step
      for run_num in range(1, num_runs + 1):
        # Concatenate the performance outcomes for all runs
        nvsone_sim_step_values = nvsone_sim_step_values + self.nvsone_matrix[(i * run_num) - 1, 1:].tolist()

      y_low.append(np.percentile(nvsone_sim_step_values, 25))
      y_median.append(np.median(nvsone_sim_step_values))
      y_high.append(np.percentile(nvsone_sim_step_values, 75))

    # Flatten matrices, turn into lists.
    leaveoneout_values = self.leaveoneout_matrix.ravel().tolist()[0]

    y_low.append(np.percentile(leaveoneout_values, 25))
    y_median.append(np.median(leaveoneout_values))
    y_high.append(np.percentile(leaveoneout_values, 75))

    x_labels = ['%s' % (item) for item in nvsone_labels]

    x_labels.append('49')
    x_range = range(0, len(x_labels))

    plt.figure()
    plt.plot(y_median, marker = 'o', label = 'Median', color = 'blue')
    plt.plot(y_high, color = 'blue', linewidth = 0.2)
    plt.plot(y_low, color = 'blue', linewidth = 0.2)
    plt.fill_between(x_range, y_high, y_low, alpha = 0.1, color = 'blue', label = '$25^{th}$ to $75^{th}$ percentile')

    plt.xticks(x_range, x_labels)
    plt.xlabel('Number of reviews in training set')
    plt.ylim(0, 1)
    plt.ylabel('WSS@95')
    plt.legend()

    plt.tight_layout()
    plt.savefig(config.PLOT_LOCATION + '/mean_compare_plot.pdf')

    font = {'size': 12}

    matplotlib.rc('font', **font)


  def mean_compare_including_random(self):
    font = {'size': 16}

    matplotlib.rc('font', **font)

    y_sim = []
    y_sim_low = []
    y_sim_median = []
    y_sim_high = []
    y_rand = []
    y_rand_low = []
    y_rand_median = []
    y_rand_high = []

    nvsone_labels = config.SIMILARITY_STEPS
    nvsone_random_labels = [str(i) for i in config.SIMILARITY_STEPS]
    num_runs = int(self.nvsone_matrix.shape[0] / len(nvsone_labels))

    # Loop over the similarity steps (i.e. training sizes)
    for i in range(1, len(nvsone_labels) + 1):
      nvsone_sim_step_values = []

      # Loop over all runs for this similarity step
      for run_num in range(1, num_runs + 1):
        # Concatenate the performance outcomes for all runs
        nvsone_sim_step_values = nvsone_sim_step_values + self.nvsone_matrix[(i * run_num) - 1, 1:].tolist()

      y_sim.append(nvsone_sim_step_values)
      y_sim_low.append(np.percentile(nvsone_sim_step_values, 25))
      y_sim_median.append(np.median(nvsone_sim_step_values))
      y_sim_high.append(np.percentile(nvsone_sim_step_values, 75))

      rand_values = self.nvsone_random_matrix[(i - 1), 1:].tolist()

      y_rand.append(rand_values)
      y_rand_low.append(np.percentile(rand_values, 25))
      y_rand_median.append(np.median(rand_values))
      y_rand_high.append(np.percentile(rand_values, 75))

    # Flatten matrices, turn into lists.
    leaveoneout_values = self.leaveoneout_matrix.ravel().tolist()[0]

    y1 = y_rand
    y2 = y_sim
    y3 = [leaveoneout_values]

    x_labels = [str(i) for i in config.SIMILARITY_STEPS]

    #### SIGNIFICANCE

    row_labels = x_labels.copy()
    random_header = ''

    all_significance_arr = list()
    random_significance_arr = list()

    for i in range(0, len(y2)):
      random_data = y1[i]
      similar_data = y2[i]
      all_data = y3[0]

      row_labels[i] = row_labels[i] + ' (mean = %f)' % np.mean(similar_data)
      random_header = random_header + 'size: %s (mean = %f)' % (x_labels[i], np.mean(random_data))

      all_p_val = stats.ranksums(similar_data, all_data).pvalue
      random_p_val = stats.ranksums(similar_data, random_data).pvalue

      all_significance_arr.append(all_p_val)
      random_significance_arr.append(random_p_val)

    all_adjusted = multi.multipletests(all_significance_arr, method = 'bonferroni')[1]
    random_adjusted = multi.multipletests(random_significance_arr, method = 'bonferroni')[1]

    df = pd.DataFrame(all_adjusted, columns = ['49 (mean = %f)' % np.mean(y3)], index = row_labels)
    df.to_excel(config.OUTPUT_LOCATION + '/significance_testing/similar_to_all_significance.xlsx', engine = 'openpyxl')

    df = pd.DataFrame(random_adjusted, columns = ['RAND ' + random_header], index = row_labels)
    df.to_excel(config.OUTPUT_LOCATION + '/significance_testing/similar_to_random_significance.xlsx', engine = 'openpyxl')

    #### Merge leave one out

    y_rand_low.append(np.percentile(leaveoneout_values, 25))
    y_rand_median.append(np.median(leaveoneout_values))
    y_rand_high.append(np.percentile(leaveoneout_values, 75))

    y_sim_low.append(np.percentile(leaveoneout_values, 25))
    y_sim_median.append(np.median(leaveoneout_values))
    y_sim_high.append(np.percentile(leaveoneout_values, 75))

    #### PLOT

    x_labels.append('49')
    x_range = range(0, len(x_labels))

    plt.figure()

    plt.plot(y_sim_median, marker = 'o', label = 'Similar median', color = 'blue')
    plt.plot(y_sim_high, color = 'blue', linewidth = 0.2)
    plt.plot(y_sim_low, color = 'blue', linewidth = 0.2)
    plt.fill_between(x_range, y_sim_high, y_sim_low, alpha = 0.1, color = 'blue')

    plt.plot(y_rand_median, marker = 's', label = 'Random median', color = 'red')
    plt.plot(y_rand_high, color = 'red', linewidth = 0.2)
    plt.plot(y_rand_low, color = 'red', linewidth = 0.2)
    plt.fill_between(x_range, y_rand_high, y_rand_low, alpha = 0.1, color = 'red')

    plt.xticks(x_range, x_labels)
    plt.xlabel('Number of reviews in training set')
    plt.ylim(0, 1)
    plt.ylabel('WSS@95')
    plt.legend()

    plt.tight_layout()
    plt.savefig(config.PLOT_LOCATION + '/mean_random_compare_plot.pdf')

    #### SEPARATE PLOT

    fig, (ax1, ax2) = plt.subplots(1, 2, sharey = True, figsize = (10, 5), gridspec_kw = {'width_ratios': [1, 1]})

    ax1.plot(y_rand_median, marker = 'o', label = 'Median', color = 'red')
    ax1.plot(y_rand_high, color = 'red', linewidth = 0.2)
    ax1.plot(y_rand_low, color = 'red', linewidth = 0.2)
    ax1.fill_between(x_range, y_rand_high, y_rand_low, alpha = 0.1, color = 'red', label = '$25^{th}$ to $75^{th}$ percentile')

    ax1.set_title('Random')
    ax1.set_xticks(x_range)
    ax1.set_xticklabels(x_labels)

    ax2.plot(y_sim_median, marker = 'o', label = 'Median', color = 'blue')
    ax2.plot(y_sim_high, color = 'blue', linewidth = 0.2)
    ax2.plot(y_sim_low, color = 'blue', linewidth = 0.2)
    ax2.fill_between(x_range, y_sim_high, y_sim_low, alpha = 0.1, color = 'blue', label = '$25^{th}$ to $75^{th}$ percentile')

    ax2.set_title('Similar')
    ax2.set_xticks(x_range)
    ax2.set_xticklabels(x_labels)

    ax1.set_ylim([0, 1])
    ax2.set_ylim([0, 1])
    ax1.legend()
    ax2.legend()

    fig.add_subplot(111, frameon = False)

    plt.tick_params(labelcolor = 'none', top = False, bottom = False, left = False, right = False)
    plt.grid(False)
    plt.xlabel('Number of reviews in training set')
    plt.ylabel('WSS@95')

    plt.tight_layout()
    plt.savefig(config.PLOT_LOCATION + '/mean_random_compare_plot_separate.pdf')

    font = {'size': 12}

    matplotlib.rc('font', **font)


  def overlay_compare(self, name_add = ''):
    # Calculate and sort the medians per dataset
    sorted_medians = np.median(self.leaveoneout_matrix, axis = 0).argsort().tolist()[0]

    self.overlay_compare_sorted('performance', sorted_medians, name_add)

    sorted_similarity = sim_fetch.get_sorted_indexes(self.similarity_matrix)

    self.overlay_compare_sorted('similarity', sorted_similarity, name_add)


  def overlay_compare_sorted(self, plot_type, sorting_idx, name_add):
    nvsone_labels = config.SIMILARITY_STEPS
    clean_nvsone_matrix = self.nvsone_matrix[:, 1:]

    # Sort the review names
    sorted_names = np.array(self.review_names)[sorting_idx]

    # Sort the data matrices
    sorted_leaveoneout_data = self.leaveoneout_matrix[:, sorting_idx]
    sorted_nvsone_data = clean_nvsone_matrix[:, sorting_idx]

    x_ticks = np.arange(1, len(sorted_names) + 1)

    fig, ax = plt.subplots()
    # Add a zero line
    plt.axhline(y = 0, alpha = 0.4, color = 'gray', linewidth = 1, linestyle = 'dashed')

    # Plot the baseline (leave-one-out) data
    self.plot_leaveoneout_overlay(plot_type, x_ticks, sorted_leaveoneout_data, sorting_idx)
    # Plot the N-vs-one data
    legend_elements = self.plot_nvsone_overlay(x_ticks, nvsone_labels, sorted_nvsone_data)

    # Add the legend element for the baseline data
    legend_elements.append(Line2D([0], [0], marker = 'o', color = 'w', markerfacecolor = 'k', alpha = 0.3, label = '49 (Baseline)'))

    extra_label = 'Mean Performance'

    if plot_type == 'similarity':
      extra_label = 'Mean Similarity'

    legend_elements.append(Line2D([0], [0], color = 'k', alpha = 0.6, linestyle = '--', label = extra_label))

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
    maxsize = max([t.get_window_extent().width for t in tl])

    # Calculate new width and adjust the plot
    added_margin = 1
    new_tick_width = maxsize / plt.gcf().dpi * len(x_ticks) + 2 * added_margin
    plt.gcf().set_size_inches(new_tick_width, plt.gcf().get_size_inches()[1])

    ax.legend(handles = legend_elements)
    plt.tight_layout()

    if plot_type == 'performance':
      name_add = name_add + '_performance_sort'
    else:
      name_add = name_add + '_similarity_sort'

    plt.savefig(config.PLOT_LOCATION + '/overlay_plot%s.pdf' % name_add)


  def prepare_data(self, sorted_data):
    # Split into reviews
    review_results = np.split(sorted_data, sorted_data.shape[1], axis = 1)

    # Turn the splits into lists
    review_results = [row.tolist() for row in review_results]
    # Flatten lists
    review_results = [[item for sublist in row for item in sublist] for row in review_results]

    return review_results


  def plot_leaveoneout_overlay(self, plot_type, x_ticks, sorted_data, sorting_idx):
    review_results = self.prepare_data(sorted_data)

    for i in x_ticks:
      y_points = review_results[i - 1]
      # Add jitter so that the points are not on
      # a straight line in the plot
      x = np.random.normal(i, 0.04, size = len(y_points))

      plt.plot(x, y_points, 'k.', alpha = 0.3, color = 'black')

    if plot_type == 'performance':
      data_matrix = np.matrix(sorted_data)

      mean_arr = data_matrix.mean(axis = 0).tolist()[0]
      min_arr = data_matrix.min(axis = 0).tolist()[0]
      max_arr = data_matrix.max(axis = 0).tolist()[0]

      plt.plot(x_ticks, mean_arr, alpha = 0.6, linestyle = 'dashed', color = 'black')
      plt.fill_between(x_ticks, min_arr, max_arr, alpha = 0.2, color = 'black')
    else:
      data_matrix = np.matrix(self.similarity_matrix)

      mean_arr = np.ravel(data_matrix.mean(axis = 1)[sorting_idx, :].tolist())
      min_arr = np.ravel(data_matrix.min(axis = 1)[sorting_idx, :].tolist())
      max_arr = np.ravel(data_matrix.max(axis = 1)[sorting_idx, :].tolist())

      plt.plot(x_ticks, mean_arr, alpha = 0.6, linestyle = 'dashed', color = 'purple')
      plt.fill_between(x_ticks, min_arr, max_arr, alpha = 0.2, color = 'purple')


  def plot_nvsone_overlay(self, x_ticks, labels, sorted_data):
    # Split into training size
    runs = np.split(sorted_data, sorted_data.shape[0] / len(config.SIMILARITY_STEPS), axis = 0)

    run_count = 0
    legend_elements = []

    for run in runs:
      run_count = run_count + 1
      run = np.split(run, run.shape[0], axis = 0)

      for i in range(0, len(config.SIMILARITY_STEPS)):
        review_results = self.prepare_data(run[i])

        color_num = self.colors[i]

        for j in x_ticks:
          y_points = review_results[j - 1]
          # Add jitter so that the points are not on
          # a straight line in the plot
          x = np.random.normal(j, 0.04, size = len(y_points))

          plt.plot(x, y_points, color = 'C%i' % color_num, marker = '.', label = labels[i])

        # Add legend element
        if run_count < 2:
          legend_elements.append(Line2D([0], [0], marker = 'o', color = 'w', markerfacecolor = 'C%i' % color_num, label = labels[i]))

    return legend_elements


  def wss_against_similarity(self):
    nvsone_labels = config.SIMILARITY_STEPS
    num_runs = int(self.nvsone_matrix.shape[0] / len(nvsone_labels))

    fig, ax = plt.subplots()

    legend_elements = []

    # Loop over all runs for this similarity step
    for i in range(0, len(nvsone_labels)):
      color_num = self.colors[i]

      subset = range(i, self.nvsone_matrix.shape[0], len(nvsone_labels))

      size_mean_wss = self.nvsone_matrix[subset, 1:].mean(axis = 0)
      size_mean_similarity = self.run_similarities[subset, 1:].mean(axis = 0)

      plt.plot(size_mean_similarity, size_mean_wss, '.', color = 'C%i' % color_num, label = nvsone_labels[i])

      legend_elements.append(Line2D([0], [0], marker = 'o', color = 'w', markerfacecolor = 'C%i' % color_num, label = nvsone_labels[i]))

    plt.xlim(0, 1.1)
    plt.xlabel('Cosine Similarity')
    plt.ylim(-0.2, 1)
    plt.ylabel('WSS@95')

    ax.legend(handles = legend_elements)
    plt.tight_layout()

    plt.savefig(config.PLOT_LOCATION + '/wss_against_similarity_plot.pdf')


  def training_selection_distribution(self):
    reviews_selected = []

    num_datasets = self.nvsone_matrix.shape[1]

    for i in range(1, num_datasets):
      dat = file_handle.load_external_classifier('n_vs_one', 1, i, 'tfidf', None)

      prev_set_len = 0

      for j in range(0, len(dat['sets'])):
        subset = dat['sets'][j]['train_reviews']
        subset = subset[prev_set_len:len(subset)]

        reviews_selected = reviews_selected + subset

        prev_set_len = config.SIMILARITY_STEPS[j]

    counted_reviews = Counter(reviews_selected)

    review_names = self.similarity_matrix.columns

    existing_reviews = counted_reviews.keys()

    remaining_reviews = set(review_names) - set(existing_reviews)

    for item in remaining_reviews:
      counted_reviews[item] = 0

    sorted_data = sorted(counted_reviews.items(), key = lambda x: x[1], reverse = True)
    sorted_values = [i[1] for i in sorted_data]
    sorted_keys = [i[0] for i in sorted_data]

    x = range(1,len(sorted_values) + 1)
    y = sorted_values

    plt.figure()

    plt.bar(x, y)

    plt.xticks(x, sorted_keys, rotation = 90)
    plt.xlabel('Review')
    plt.ylabel('Times in training set\n(Total # of sets = %i)' % ((num_datasets - 1) * len(config.SIMILARITY_STEPS)))

    # Adjust the left and right margins
    plt.gca().margins(x = 0.01)
    # Create a canvas
    plt.gcf().canvas.draw()
    # Get the widest tick label
    tl = plt.gca().get_xticklabels()
    maxsize = max([t.get_window_extent().width for t in tl])

    # Calculate new width and adjust the plot
    added_margin = 1
    new_tick_width = maxsize / plt.gcf().dpi * len(x) + 2 * added_margin
    plt.gcf().set_size_inches(new_tick_width, plt.gcf().get_size_inches()[1])

    plt.tight_layout()

    plt.savefig(config.PLOT_LOCATION + '/training_set_distribution.pdf')


  def group_split_means(self):
    font = {'size': 13}

    self.run_similarities

    matplotlib.rc('font', **font)

    sorting_idx = np.median(self.leaveoneout_matrix, axis = 0).argsort().tolist()[0]

    nvsone_labels = config.SIMILARITY_STEPS

    clean_nvsone_matrix = self.nvsone_matrix[:, 1:]
    num_runs = int(self.nvsone_matrix.shape[0] / len(nvsone_labels))

    similarity_array = self.run_similarities[:, 1:]

    # Sort the review names
    sorted_names = np.array(self.review_names)[sorting_idx]
    sorted_names = sorted_names.tolist()

    # Sort the data matrices
    sorted_leaveoneout_data = np.array(self.leaveoneout_matrix[:, sorting_idx])
    sorted_nvsone_data = clean_nvsone_matrix[:, sorting_idx]
    sorted_similarity_data = similarity_array[:, sorting_idx]

    # Merge leaveoneout into nvsone
    data_array = np.concatenate((sorted_nvsone_data, sorted_leaveoneout_data))

    x_labels = [str(i) for i in config.SIMILARITY_STEPS] + ['49']
    x_range = range(0, len(x_labels))

    def summarise_lines(data):
      size_data_low = []
      size_data_median = []
      size_data_high = []

      # Loop over the similarity steps (i.e. training sizes)
      for i in range(1, len(x_labels) + 1):
        nvsone_sim_step_values = []

        # Concatenate the performance outcomes for all runs
        nvsone_sim_step_values = nvsone_sim_step_values + data[i - 1, :].tolist()

        size_data_low.append(np.percentile(nvsone_sim_step_values, 25))
        size_data_median.append(np.median(nvsone_sim_step_values))
        size_data_high.append(np.percentile(nvsone_sim_step_values, 75))

      return size_data_low, size_data_median, size_data_high

    for group_type, group_reviews in config.GROUPS.items():
      group_idx = [sorted_names.index(x) for x in group_reviews]

      group_data = data_array[:, group_idx]
      group_similarity = sorted_similarity_data[:, group_idx]
      mean_similarity_per_set_size = group_similarity.mean(axis = 1).tolist()

      mean_similarity_per_set_size.append(0)

      y_low, y_median, y_high = summarise_lines(group_data)

      plt.figure()

      plt.plot(y_median, marker = 's', label = '1 - 7 median', color = 'blue')
      plt.plot(y_high, color = 'blue', linewidth = 0.2)
      plt.plot(y_low, color = 'blue', linewidth = 0.2)
      plt.fill_between(x_range, y_high, y_low, alpha = 0.1, color = 'blue', label = '$25^{th}$ to $75^{th}$ percentile')

      min_y = 0

      if min(y_median) < 0:
        min_y = -0.2

      plt.xticks(x_range, x_labels)
      plt.xlabel('Number of reviews in training set')
      plt.ylim(min_y, 1)
      plt.ylabel('WSS@95')
      plt.title(group_type)
      plt.legend()

      plt.tight_layout()
      plt.savefig(config.PLOT_LOCATION + '/review_groups/group_%s.pdf' % group_type)

    font = {'size': 12}

    matplotlib.rc('font', **font)
