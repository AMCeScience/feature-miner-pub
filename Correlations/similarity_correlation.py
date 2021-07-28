from sklearn.metrics.pairwise import cosine_similarity
import Libs.outcome_fetcher as fetcher
import Libs.file_storage as file_handle
import Libs.outcome_fetcher as fetcher
import Database.db_connector as db
import pandas as pd, numpy as np, config, os

def calculate():
  external_data = load_data()

  # Loop over the similarity step sizes
  for size in config.SIMILARITY_STEPS:
    # Get the matrix from which we can calculate correlations
    # Pass in the external data tuple by unpacking
    df = build_matrix(size, *external_data)

    # Calculate correlation between all columns
    correlations = df.corr()

    # Store correlation matrix as CSV file
    filename = config.CORRELATION_LOCATION + '/similarity_calculation_correlation_matrix_size_%i.xlsx' % size

    if not os.path.exists(os.path.dirname(filename)):
      os.makedirs(os.path.dirname(filename))

    correlations.to_excel(filename)


def load_data():
  conn = db.Connector()
  labels = conn.get_review_names()
  review_indices = conn.get_review_indices()

  # Fetch the cosine matrix
  cos_sim = file_handle.load_matrix('cosine_sim_matrix')
  # Fetch the mean TF-IDF values for each review
  mean_review_matrix = get_mean_tfidf_matrix(review_indices)

  # Fetch the WSS outcomes for the n-vs-one experiment
  outcome_fetcher = fetcher.Outcome_fetcher()
  nvsone_matrix, _, _, _ = outcome_fetcher.get_data('n_vs_one')

  sizes = config.SIMILARITY_STEPS
  num_runs = int(nvsone_matrix.shape[0] / len(sizes))

  mean_wss = {}

  # Find the mean WSS for the n-vs-one experiment
  for i in range(0, len(sizes)):
    nvsone_mean = []

    # Get the row indexes for the specified size
    # When there are 5 runs and 4 similarity steps
    # the range produces: 0, 5, 10, 15 for i = 0
    rows = range(i, nvsone_matrix.shape[0], num_runs)

    # Get the mean of the selected rows
    # and add to the mean_wss dictionary
    mean_wss[sizes[i]] = nvsone_matrix[rows, 1:].mean(axis = 0)

  # Return the fetched data
  return labels, cos_sim, mean_review_matrix, mean_wss


def build_matrix(size, labels, cos_sim, mean_review_matrix, mean_wss):
  correlation_matrix = None

  # Loop over the reviews
  for review in cos_sim.index:
    # Get the similarity data for the specified size
    sim_obj = get_similarities(review, size, cos_sim, labels)

    # Get the cosine similarity for the full set
    set_sim = get_combined_set_similarity(mean_review_matrix, sim_obj)

    # Get the cosine similarity for the optimal set
    # optimal_sim = get_optimal_set_similarity(size, sim_obj)
    
    # Build the matrix row (i.e. vector)
    vector = [set_sim, sim_obj['median'], sim_obj['mean'], sim_obj['minimum'], sim_obj['maximum'], mean_wss[size][sim_obj['review_idx']]]

    # Build the matrix from which we can calculate correlations
    if correlation_matrix is None:
      correlation_matrix = np.matrix(vector)
    else:
      correlation_matrix = np.vstack([correlation_matrix, vector])

  # Convert into pandas dataframe so we can add column headers
  correlation_matrix = pd.DataFrame(correlation_matrix, columns = ['combined set similarity', 'median similarity', 'mean similarity', 'minimum similarity', 'maximum similarity', 'WSS'])

  return correlation_matrix


def get_optimal_set_similarity(size, sim_obj):
  # Load the combination sets and their similarities
  combination_set = file_handle.load_combination_set(size)

  # For size = 10 no combination set exists, so check here
  if combination_set is not None:
    # Get a list of similarities
    sims = [i['similarity'] for i in combination_set]

    # Return the cosine similarity of the review
    return sims[sim_obj['review_idx']]

  # For size = 10 return 0, corr() function will blank
  # this value out in the final correlation matrix
  return 0


def get_combined_set_similarity(mean_review_matrix, sim_obj):
  # Get the subset of included reviews
  included_reviews = mean_review_matrix[sim_obj['subset_idx']]
  # Get the vector of the test review
  this_review = mean_review_matrix[sim_obj['review_idx']]

  # Get the mean of the included reviews subset
  review_mean = np.mean(included_reviews, axis = 0)

  # Build matrix
  sim_matrix = np.vstack([this_review, review_mean])

  # Calculate cosine similarity and return
  return cosine_similarity(sim_matrix)[0, 1]


def get_similarities(review, size, cos_sim, labels):
  # Get the index of the review
  review_idx = labels.index(review)

  # Get the similarities for the specified review
  similarities = cos_sim.loc[[review]]

  # Drop its own similarity (i.e. 1.0)
  similarities = similarities.drop(review, axis = 1)
  # Sort similarities high to low
  similarities = similarities.sort_values(by = review, ascending = False, axis = 1)

  # Get the subset for the specified size
  sim_subset = similarities.iloc[:,0:size]

  # Find the review indexes of the subset
  subset_reviews = sim_subset.columns
  subset_idx = [labels.index(i) for i in subset_reviews]
  
  # Build object and return
  return {
    'review': review,
    'review_idx': review_idx,
    'subset_idx': subset_idx,
    'median': sim_subset.median(axis = 1),
    'mean': sim_subset.mean(axis = 1),
    'minimum': sim_subset.min(axis = 1),
    'maximum': sim_subset.max(axis = 1)
  }


def get_mean_tfidf_matrix(review_indices):
  tfidf_matrix = file_handle.load_matrix('tfidf_matrix')

  mean_review_matrix = None

  # Build a mean review TF matrix
  for i in range(0, len(review_indices)):
    l,r = review_indices[i]

    # Get the subset of review documents
    review_subset = tfidf_matrix[l:r,:]

    # Get the mean vector for this review
    review_mean = np.mean(review_subset, axis = 0)

    # Build matrix
    if mean_review_matrix is None:
      mean_review_matrix = np.matrix(review_mean)
    else:
      mean_review_matrix = np.vstack([mean_review_matrix, review_mean])

  return mean_review_matrix
