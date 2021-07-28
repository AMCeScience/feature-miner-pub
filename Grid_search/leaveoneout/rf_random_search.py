import os, sys
root_path = os.path.abspath('')
sys.path.append(root_path + '/')

import config
import csv
import Libs.performance_calculation as performance
import Libs.file_storage as file_handle
import Libs.create_fold as create_fold
from sklearn.ensemble import RandomForestClassifier
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

    best_params = search_random_forest(X_train, y_train)

    if file_handle.file_exists(root_path + '/Grid_search/leaveoneout/random_search.csv') is False:
      with open(root_path + '/Grid_search/leaveoneout/random_search.csv', 'a') as handle:
        writer = csv.DictWriter(handle, best_params.keys())
        writer.writeheader()

    with open(root_path + '/Grid_search/leaveoneout/random_search.csv', 'a') as handle:
      writer = csv.DictWriter(handle, best_params.keys())
      writer.writerow(best_params)


def search_random_forest(X, y):
  random_grid = {
    'bootstrap': [True, False],
    'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110],
    'max_features': ['log2', 'sqrt'],
    'min_samples_leaf': [1, 2, 4],
    'min_samples_split': [2, 5, 7, 10],
    'n_estimators': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
  }

  score = make_scorer(performance.wss, greater_is_better = True)

  clf = RandomForestClassifier()

  rf_random = RandomizedSearchCV(estimator = clf, param_distributions = random_grid, n_iter = 100, cv = 3, n_jobs = config.NUM_JOBS, scoring = score)

  rf_random.fit(X, y)

  return rf_random.best_params_


def unpack_data(data):
  return (data['X_train'], data['y_train'])


if __name__ == "__main__":
  run_tf()
  run_tm()
