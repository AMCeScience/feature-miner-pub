import Libs.similarity_fetcher as sim_fetch
import Libs.file_storage as file_handle
import os, config
import numpy as np

class Outcome_fetcher(object):

  outcome_type = ''
  num_runs = None
  num_dataset = None
  subset_names = None


  def get_data(self, outcome_type, subset_names = None):
    self.outcome_type = outcome_type

    if subset_names is not None:
      self.subset_names = subset_names

    self.num_runs = self.count_runs()
    self.num_dataset = self.count_datasets()

    if self.outcome_type == 'leave_one_out':
      return self.get_leaveoneout_data()

    if self.outcome_type == 'one_vs_one':
      return self.get_onevsone_data()

    if self.outcome_type == 'n_vs_one':
      return self.get_nvsone_data()

    if self.outcome_type == 'n_vs_one_random':
      return self.get_nvsone_random_data()


  def get_timing_data(self, outcome_type):
    self.outcome_type = outcome_type

    self.num_runs = self.count_runs()
    self.num_dataset = self.count_datasets()

    if self.outcome_type == 'leave_one_out':
      return self.get_leaveoneout_timing()

    if self.outcome_type == 'n_vs_one':
      return self.get_nvsone_timing()


  def count_runs(self):
    folders = [name for name in os.listdir(config.OUTPUT_LOCATION + '/' + self.outcome_type) if 'run_' in name]

    runs = len(folders)

    if runs == 0:
      return 1

    return runs


  def count_datasets(self):
    # Recursive file count
    files = [name for dp, dn, fn in os.walk(config.OUTPUT_LOCATION + '/%s/' % self.outcome_type) for name in fn if 'classifier.pickle' in name]

    return len(files)


  def get_onevsone_data(self):
    result_list = []

    # Loop over the datasets
    for i in range(1, self.num_dataset + 1):
      result = file_handle.load_external_classifier('one_vs_one', 1, i, 'tfidf', None)

      # Retrieve the WSS values from the data dictionary
      result_vect = [i['wss_95'] for i in result['outcomes']]

      # Build a list of WSS values per review
      result_list.append(result_vect)

    # Stack the WSS lists into a matrix
    return np.vstack(result_list)


  def get_nvsone_timing(self):
    data_list = []

    # Loop over the runs
    for run_num in range(1, self.num_runs + 1):
      data_list.append(self.get_nvsone_run_timing(run_num))

    return data_list


  def get_nvsone_run_timing(self, num_run):
    results = []

    # Loop over the datasets
    for i in range(1, self.num_dataset + 1):
      result = file_handle.load_external_classifier('n_vs_one', num_run, i, 'tfidf', None)

      if self.subset_names is not None and result['test_review'] not in self.subset_names:
        continue

      results.append([i['train_time'] for i in result['sets']])

    return results


  def get_nvsone_data(self):
    nvsone_matrix = None
    similarity_matrix = None

    # Loop over the runs
    for run_num in range(1, self.num_runs + 1):
      run_matrix, run_similarity_matrix, test_review_names = self.get_nvsone_run(run_num)
      run_matrix = run_matrix.transpose()
      run_similarity_matrix = run_similarity_matrix.transpose()

      if nvsone_matrix is None:
        nvsone_matrix = run_matrix
        similarity_matrix = run_similarity_matrix
      else:
        # Stack the retrieved values with the existing matrix
        nvsone_matrix = np.vstack([nvsone_matrix, run_matrix])
        similarity_matrix = np.vstack([similarity_matrix, run_similarity_matrix])

    outcome_fetcher = Outcome_fetcher()
    leaveoneout_matrix = outcome_fetcher.get_data('leave_one_out', test_review_names)

    def sect(x):
      # Get the first cell of a apply_along_axis value.
      first_cell = x.tolist()[0][0]

      # Check whether the cell contains the string in 'feature'.
      return 'TF - rf' == first_cell

    # Retrieve all rows with TF outcomes.
    rows = np.apply_along_axis(sect, axis = 1, arr = leaveoneout_matrix)

    # Remove first column containing identifiers
    # and turn all data into floats
    clean_leaveoneout_matrix = leaveoneout_matrix[rows, 1:].astype(float)

    return nvsone_matrix, clean_leaveoneout_matrix, similarity_matrix, test_review_names


  def get_nvsone_run(self, num_run):
    cos_sim = sim_fetch.get_similarity()

    result_matrix = np.array(config.SIMILARITY_STEPS)
    similarity_matrix = np.array(config.SIMILARITY_STEPS)

    review_names = []

    # Loop over the datasets
    for i in range(1, self.num_dataset + 1):
      result = file_handle.load_external_classifier('n_vs_one', num_run, i, 'tfidf', None)

      if self.subset_names is not None and result['test_review'] not in self.subset_names:
        continue

      review_names.append(result['test_review'])

      outcome_sets = result['sets']

      result_vect = [i['outcomes'][0]['wss_95'] for i in outcome_sets]

      result_matrix = np.vstack([result_matrix, result_vect])

      sim_vect = [cos_sim.loc[result['test_review'], i['train_reviews']].mean() for i in outcome_sets]

      similarity_matrix = np.vstack([similarity_matrix, sim_vect])

    # Stack the WSS lists into a matrix
    return np.vstack(result_matrix), np.vstack(similarity_matrix), review_names


  def get_nvsone_random_data(self):
    nvsone_matrix = None

    # Loop over the runs
    run_matrix = self.get_nvsone_random_run()
    nvsone_matrix = run_matrix.transpose()

    return nvsone_matrix


  def get_nvsone_random_run(self):
    cos_sim = sim_fetch.get_similarity()

    result_matrix = np.array(config.SIMILARITY_STEPS)

    # Loop over the datasets
    for i in range(1, self.num_dataset + 1):
      result = file_handle.load_external_classifier('n_vs_one_random', 1, i, 'tfidf', None)

      outcome_sets = result['sets']

      result_vect = [i['outcomes'][0]['wss_95'] for i in outcome_sets]

      result_matrix = np.vstack([result_matrix, result_vect])

    # Stack the WSS lists into a matrix
    return np.vstack(result_matrix)


  def get_leaveoneout_timing(self):
    data_list = []

    # Loop over the runs
    for run_num in range(1, self.num_runs + 1):
      data_list.append(self.get_leaveoneout_run_timing(run_num))

    return data_list


  def get_leaveoneout_run_timing(self, num_run, tm_set = None):
    classifier_type = 'TF'
    dataset_type = 'tfidf'
    rf_timing_list = []

    # Loop over the TF datasets
    for i in range(1, self.num_dataset + 1):
      fold_data = file_handle.load_external_classifier('leave_one_out', num_run, i, dataset_type, tm_set)

      if self.subset_names is not None and fold_data['test_review_names'][0] not in self.subset_names:
        continue

      if 'rf' in fold_data:
        rf_timing_list.append(fold_data['rf']['rf_train_time'])

    return rf_timing_list


  def get_leaveoneout_data(self):
    data_matrix = None

    # Loop over the runs
    for run_num in range(1, self.num_runs + 1):
      a, b = self.get_leaveoneout_run(run_num)

      tf_matrix = np.matrix([a, b])

      if data_matrix is None:
        # Create a data matrix
        data_matrix = tf_matrix
      else:
        # Stack the retrieved values with the existing matrix
        data_matrix = np.vstack([data_matrix, tf_matrix])

      # Loop over the TMs
      for tm_set in range(1, config.NUM_TOPICMODELS + 1):
        a, b = self.get_leaveoneout_run(run_num, tm_set)

        # Stack the retrieved values with the existing matrix
        data_matrix = np.vstack([data_matrix, a, b])

    return data_matrix


  def get_leaveoneout_run(self, num_run, tm_set = None):
    classifier_type = 'TF'
    dataset_type = 'tfidf'

    if tm_set is not None:
      classifier_type = 'TM'
      dataset_type = 'tm'

    # Add an identifier label to the list
    rf_all_wss = ['%s - rf' % classifier_type]
    svm_all_wss = ['%s - svm' % classifier_type]

    # Loop over the TF datasets
    for i in range(1, self.num_dataset + 1):
      fold_data = file_handle.load_external_classifier('leave_one_out', num_run, i, dataset_type, tm_set)

      if self.subset_names is not None and fold_data['test_review_names'][0] not in self.subset_names:
        continue

      # Retrieve the WSS values from the data dictionary
      rf, svm = self.fetch_wss(fold_data)

      # Merge the list of WSS values
      rf_all_wss = rf_all_wss + rf
      svm_all_wss = svm_all_wss + svm

    return rf_all_wss, svm_all_wss


  def fetch_wss(self, data):
    rf_wss = None
    svm_wss = None

    if 'rf' in data:
      # Unpack the RF data dictionary
      rf_outcomes = data['rf']['outcomes']
      # Put all the WSS@95 values into a list
      rf_wss = [x['wss_95'] for x in rf_outcomes]

    if 'svm' in data:
      # Unpack the RF data dictionary
      svm_outcomes = data['svm']['outcomes']
      # Put all the WSS@95 values into a list
      svm_wss = [x['wss_95'] for x in svm_outcomes]

    if rf_wss is None:
      return svm_wss

    if svm_wss is None:
      return rf_wss

    return rf_wss, svm_wss
