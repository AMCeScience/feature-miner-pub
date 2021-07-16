import time
import Libs.file_storage as outcomes_handle
import Libs.performance_calculation as performance
import Libs.run_model as run_model
import config
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

def run_models(data, fold):
  X_train, y_train, X_test, y_test, test_indices, test_review_names = outcomes_handle.unpack_dataset(data)

  start_time = time.time()

  rf_model = fit_random_forest(X_train, y_train)
  rf_outcomes = performance.test_classifier(rf_model, X_test, y_test, test_indices, test_review_names)

  rf_elapsed_time = time.time() - start_time

  svm_model = fit_svm(X_train, y_train)
  svm_outcomes = performance.test_classifier(svm_model, X_test, y_test, test_indices, test_review_names)

  svm_elapsed_time = time.time() - start_time

  classifier_obj = {
    'rf': {
      'rf_train_time': rf_elapsed_time,
      'outcomes': rf_outcomes
    },
    'svm': {
      'svm_train_time': svm_elapsed_time,
      'outcomes': svm_outcomes
    },
    'test_review_names': test_review_names
  }

  return classifier_obj


def fit_random_forest(X, y):
  clf = RandomForestClassifier(bootstrap = True, max_depth = 100, max_features = 'log2', min_samples_leaf = 1, min_samples_split = 5, n_estimators = 500)

  clf.fit(X, y)

  return clf


def fit_svm(X, y):
  clf = SGDClassifier(penalty = 'l2', alpha = 0.0000001, max_iter = 500, tol = 0.000001, learning_rate = 'optimal', eta0 = 0.1)

  clf.fit(X, y)

  return clf


def classify():
  model_runner = run_model.Run_model(run_models, 'leave_one_out', run_parallel = True)

  model_runner.run_tf()
  model_runner.run_tm()


if __name__ == "__main__":
  classify()
