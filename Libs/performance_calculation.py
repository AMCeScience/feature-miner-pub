import numpy as np
from sklearn.metrics import *

def test_classifier(clf, X_test, y_test, test_indices, test_review_names):
  outcomes = []

  for i in range(0, len(test_indices)):
    l = test_indices[i][0]
    r = test_indices[i][1] + 1
    
    X_subset = X_test[l:r,:]
    y_subset = y_test[l:r]
    
    pred_labs = clf.predict(X_subset)

    # Get probabilities instead of label prediction
    try:
      pred_probs = clf.decision_function(X_subset)
    except AttributeError:
      pred_probs = clf.predict_proba(X_subset)
      pred_probs = np.asarray([r for l, r in pred_probs])
    
    outcomes.append({
      'review_name': test_review_names[i],
      'confusion_matrix': confusion_matrix_format(y_subset, pred_labs),
      'wss_95': wss(y_subset, pred_probs, 0.95),
      'wss_100': wss(y_subset, pred_probs, 1.0),
      'auc': get_roc_auc(y_subset, pred_probs)
    })

  return outcomes


def wss(y_true, y_pred, recall = 0.95):
  # Count all includes
  includes_count = y_true.sum()
  
  # Get indices of includes
  includes_indices = np.nonzero(y_true)[0]

  # Get predicted probabilities of included class
  included_pred_probs = y_pred[includes_indices]

  # Calculate the number of documents to be included given the recall
  number_removed = includes_count - int(recall * includes_count)

  # Get the threshold of the probability
  if number_removed == includes_count:
    number_removed = includes_count - 1
  
  threshold_probs = np.sort(included_pred_probs)[number_removed]
  
  wss = len(y_pred[y_pred < threshold_probs]) / len(y_pred) - (1 - recall)
  
  return wss


def get_roc_auc(y_subset, pred_probs):
  if len(np.nonzero(y_subset)[0]) != len(y_subset):
    return roc_auc_score(y_subset, pred_probs)

  return None


def confusion_matrix_format(y_test, prediction):
  matrix = confusion_matrix(y_test, prediction)

  if matrix.shape[0] is 2 and matrix.shape[1] is 2:
    tn, fp, fn, tp = confusion_matrix(y_test, prediction).ravel()

    return (tn, fp, fn, tp)

  return None
