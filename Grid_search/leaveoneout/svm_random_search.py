import os, sys
root_path = os.path.abspath('')
sys.path.append(root_path)

import config
import csv
from Preprocessing import read_write_corpus
import Libs.fold_calculation as fold_calc
import Libs.performance_calculation as performance
import Libs.file_storage as file_handle
import Libs.create_fold as create_fold
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import *
import numpy as np

def run_tf():
  print('running TF-IDF')
  tfidf_matrix = file_handle.load_matrix('tfidf_matrix')

  run_folds(tfidf_matrix)


def run_tm():
  print('running TM')
  print('running a total of %i datasets' % config.NUM_TOPICMODELS)

  for dataset_num in range(1, config.NUM_TOPICMODELS + 1):
    topicmodel = file_handle.load_topicmodel(dataset_num)
    tm_matrix = topicmodel['matrix']

    run_folds(tm_matrix)


def run_folds(matrix):
  fold_creator = create_fold.Create_fold(matrix, 'leave_one_out')

  num_folds = fold_creator.determine_folds()

  print('running a total of %i folds' % num_folds)

  for fold in range(1, num_folds + 1):
    print('on fold #%i' % fold)
    data = fold_creator.get_fold(fold)

    X_train, y_train = unpack_data(data)

    best_params = search_svm(X_train, y_train)

    if file_handle.file_exists(root_path + '/Grid_search/leaveoneout/random_search.csv') is False:
      with open(root_path + '/Grid_search/leaveoneout/random_search.csv', 'a') as handle:
        writer = csv.DictWriter(handle, best_params.keys())
        writer.writeheader()

    with open(root_path + '/Grid_search/leaveoneout/random_search.csv', 'a') as handle:
      writer = csv.DictWriter(handle, best_params.keys())
      writer.writerow(best_params)


def search_svm(X, y):
  random_grid = {
    'penalty': ['l2', 'elasticnet'],
    'alpha': [0.0000001, 0.00001, 0.001, 0.1, 1.0],
    'max_iter': [10, 100, 1000],
    'tol': [0.000001, 0.001, 0.1, None],
    'learning_rate': ['optimal', 'invscaling', 'constant'],
    'class_weight': [{1: 2, 0: 1}, {1: 1, 0: 1}, {1: 5, 0: 1}],
    'eta0': [0.01, 0.1, 1.0]
  }
  
  score = make_scorer(performance.wss, greater_is_better = True)

  svm = SGDClassifier()

  svm_random = RandomizedSearchCV(estimator = svm, param_distributions = random_grid, n_iter = 100, cv = 3, n_jobs = config.NUM_JOBS, scoring = score)

  svm_random.fit(X, y)

  return svm_random.best_params_


def unpack_data(data):
  return (data['X_train'], data['y_train'])


if __name__ == "__main__":
  run_tf()
  run_tm()
