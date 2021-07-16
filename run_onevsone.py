import time
import Libs.file_storage as outcomes_handle
import Libs.performance_calculation as performance
import Libs.run_model as run_model
import config
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import *
import numpy as np

def run_models(data, fold):
  X_train, y_train, X_test, y_test, test_indices, test_review_names = outcomes_handle.unpack_dataset(data)

  train_review = test_review_names[fold - 1]

  start_time = time.time()

  rf_grid = fit_random_forest(X_train, y_train)
  rf_model = rf_grid.best_estimator_
  rf_outcomes = performance.test_classifier(rf_model, X_test, y_test, test_indices, test_review_names)

  elapsed_time = time.time() - start_time

  classifier_obj = {
    'train_time': elapsed_time,
    'train_review': train_review,
    'parameters': rf_grid.best_params_,
    'outcomes': rf_outcomes
  }

  return classifier_obj


def fit_random_forest(X, y):
  # Create the parameter grid based on the results of random search.
  param_grid = {
    'bootstrap': [True, False],
    'max_depth': [20, 50, 100],
    'max_features': ['log2', 'sqrt'],
    'min_samples_leaf': [2, 4],
    'min_samples_split': [2, 5, 10],
    'n_estimators': [100, 400, 900]
  }

  num_include = len(np.where(y == True)[0])

  # Make CV selector for one review with only 2 includes
  num_cv = 3

  if num_include < 3:
    num_cv = 2

  # Add WSS@95 scorer
  score = make_scorer(performance.wss, greater_is_better = True)

  clf = RandomForestClassifier()

  grid_search = GridSearchCV(estimator = clf, param_grid = param_grid, cv = num_cv, n_jobs = config.NUM_JOBS, scoring = score)

  grid_search.fit(X, y)

  return grid_search


def classify():
  model_runner = run_model.Run_model(run_models, 'one_vs_one')

  model_runner.run_tf()


if __name__ == "__main__":
  classify()
