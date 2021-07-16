import Libs.file_storage as file_handle
import numpy as np

def get_similarity():
  cosine_similarity = file_handle.load_matrix('cosine_sim_matrix')

  return cosine_similarity


def get_sorted_indexes(cos_sim):
  mean_sim = cos_sim.mean(axis = 1)
  
  sorted_similarity = np.matrix(mean_sim).argsort().tolist()[0]
  
  return sorted_similarity


def get_cut_matrix(review_names, remove_diagonal = True):
  cos_sim = get_similarity()

  if remove_diagonal is True:
    np.fill_diagonal(cos_sim.values, 0)

  cut_sim = cos_sim.loc[review_names]

  return cut_sim


def get_cut_names(low = True):
  cosine_similarity = get_similarity()

  # Sort all values in the cosine similarity matrix
  cosine_similarity.values.sort()
  cosine_ordered = cosine_similarity.iloc[:,::-1]
  # Get the first column of the sorted similarity matrix
  cosine_col = cosine_ordered.iloc[:, 1]
  # Calculate the median value of the first column
  first_col_median = cosine_col.median()

  if low is True:
    # Select the reviews that lie above this median
    selected_reviews = cosine_col[cosine_col > first_col_median].index.tolist()
  else:
    # Select the reviews that lie above this median
    selected_reviews = cosine_col[cosine_col < first_col_median].index.tolist()

  return selected_reviews
