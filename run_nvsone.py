import time
import Libs.file_storage as outcomes_handle
import Libs.performance_calculation as performance
import Libs.run_model as run_model
import Libs.file_storage as file_handle
import config
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import *
import numpy as np

def run_models(data, fold):
  X_train_sets, y_train_sets, X_test, y_test, test_indices, test_review_names = outcomes_handle.unpack_dataset(data)
  train_review_names = data['train_review_names']

  classifier_obj = {
    'test_review': test_review_names[0],
    'sets': []
  }

  for i in range(0, len(X_train_sets)):
    single_obj = file_handle.load_classifier_part(fold, 'tfidf', i)

    if single_obj is not None:
      classifier_obj['sets'].append(single_obj)

      continue

    print('fold #%i, on set %i of %i' % (fold, i, len(X_train_sets)))

    X_train = X_train_sets[i]
    y_train = y_train_sets[i]
    review_names = train_review_names[i]

    start_time = time.time()

    rf_grid = fit_random_forest(X_train, y_train)
    rf_model = rf_grid.best_estimator_
    rf_outcomes = performance.test_classifier(rf_model, X_test, y_test, test_indices, test_review_names)

    elapsed_time = time.time() - start_time

    single_obj = {
      'train_time': elapsed_time,
      'train_reviews': review_names,
      'parameters': rf_grid.best_params_,
      'outcomes': rf_outcomes
    }

    file_handle.store_classifier_part(fold, 'tfidf', i, single_obj)

    classifier_obj['sets'].append(single_obj)

  return classifier_obj


def fit_random_forest(X, y):
  # Create the parameter grid based on the results of random search.
  param_grid = {
    'bootstrap': [True],
    'max_depth': [20, 50, 100],
    'max_features': ['sqrt'],
    'min_samples_leaf': [2],
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
  model_runner = run_model.Run_model(run_models, 'n_vs_one')

  model_runner.run_tf()


if __name__ == "__main__":
  classify()
