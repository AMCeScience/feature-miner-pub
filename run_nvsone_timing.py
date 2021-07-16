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

    rf_model = fit_random_forest(X_train, y_train)

    elapsed_time = time.time() - start_time

    rf_outcomes = performance.test_classifier(rf_model, X_test, y_test, test_indices, test_review_names)


    single_obj = {
      'train_time': elapsed_time,
      'train_reviews': review_names,
      'outcomes': rf_outcomes
    }

    file_handle.store_classifier_part(fold, 'tfidf', i, single_obj)

    classifier_obj['sets'].append(single_obj)

  return classifier_obj


def fit_random_forest(X, y):
  clf = RandomForestClassifier(bootstrap = True, max_depth = 100, max_features = 'log2', min_samples_leaf = 1, min_samples_split = 5, n_estimators = 500)

  clf.fit(X, y)

  return clf


def classify():
  model_runner = run_model.Run_model(run_models, 'n_vs_one', run_parallel = True)

  model_runner.run_tf()


if __name__ == "__main__":
  classify()
