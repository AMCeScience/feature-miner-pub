from multiprocessing import Pool
from functools import partial
import Libs.create_fold as create_fold
import Libs.file_storage as file_handle
import config

class Run_model(object):
  experiment_type = None
  run_func = None
  run_parallel = False
  feature_type = 'tfidf'


  def __init__(self, run_func, experiment_type, run_parallel = False):
    self.run_func = run_func
    self.experiment_type = experiment_type
    self.run_parallel = run_parallel


  def run_tm(self):
    self.feature_type = 'tm'

    print('running TM')
    print('running a total of %i datasets' % config.NUM_TOPICMODELS)

    for dataset_num in range(1, config.NUM_TOPICMODELS + 1):
      topicmodel = file_handle.load_topicmodel(dataset_num)
      tm_matrix = topicmodel['matrix']

      self.run_folds(tm_matrix, dataset_num)


  def run_tf(self):
    self.feature_type = 'tfidf'

    print('running TF-IDF')
    tf_matrix = file_handle.load_matrix('tfidf_matrix')

    self.run_folds(tf_matrix)


  def run_folds(self, matrix, dataset_num = 1):
    fold_creator = create_fold.Create_fold(matrix, self.experiment_type)

    num_folds = fold_creator.determine_folds()
    print('running a total of %i folds' % num_folds)

    fold_range = range(1, num_folds + 1)

    if self.run_parallel:
      # Fix parameters of run_fold function
      parallel_func = partial(self.run_fold,
          fold_creator = fold_creator,
          run_func = self.run_func,
          feature_type = self.feature_type,
          dataset_num = dataset_num)

      with Pool(processes = config.POOL_PROCESSES) as pool:
        # Parallelise the folds
        pool.map(parallel_func, fold_range)
    else:
      for fold in fold_range:
        self.run_fold(fold, fold_creator, self.run_func, self.feature_type, dataset_num)

    print('Number of docs in matrix: %i' % fold_creator.X.shape[0])
    print('Number of docs in test sets: %i' % fold_creator.x_test_total_size)


  def run_fold(self, fold, fold_creator, run_func, feature_type, dataset_num):
    print('on fold #%i' % fold)

    classifier_obj = file_handle.load_classifier(fold, feature_type, dataset_num)

    if classifier_obj is None:
      data = fold_creator.get_fold(fold)

      classifier_obj = run_func(data, fold)

      file_handle.store_classifier(fold, feature_type, classifier_obj, dataset_num)
