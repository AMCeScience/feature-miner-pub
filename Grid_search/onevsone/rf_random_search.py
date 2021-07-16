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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import *
import numpy as np

def run_tf():
  print('running TF')
  tf_matrix = file_handle.load_matrix('tfidf_matrix')

  fold_creator = create_fold.Create_fold(tf_matrix, 'one_vs_one')

  num_folds = fold_creator.determine_folds()
  print('running a total of %i folds' % num_folds)

  for fold in range(1, num_folds + 1):
    print('on fold #%i' % fold)
    data = fold_creator.get_fold(fold)

    X_train, y_train = unpack_data(data)

    best_params = search_random_forest(X_train, y_train)

    if file_handle.file_exists(root_path + '/Grid_search/onevsone/random_search.csv') is False:
      with open(root_path + '/Grid_search/onevsone/random_search.csv', 'a') as handle:
        writer = csv.DictWriter(handle, best_params.keys())
        writer.writeheader()

    with open(root_path + '/Grid_search/onevsone/random_search.csv', 'a') as handle:
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
  
  num_include = len(np.where(y == True)[0])

  num_cv = 3

  if num_include < 3:
    num_cv = 2

  score = make_scorer(performance.wss, greater_is_better = True)

  clf = RandomForestClassifier()

  rf_random = RandomizedSearchCV(estimator = clf, param_distributions = random_grid, n_iter = 250, cv = num_cv, n_jobs = config.NUM_JOBS, scoring = score)

  rf_random.fit(X, y)

  return rf_random.best_params_


def unpack_data(data):
  return (data['X_train'], data['y_train'])


if __name__ == "__main__":
  run_tf()
