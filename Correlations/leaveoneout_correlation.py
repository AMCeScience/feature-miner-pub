from sklearn.metrics.pairwise import cosine_similarity
import Libs.file_storage as file_handle
import Libs.outcome_fetcher as fetcher
import Database.db_connector as db
import pandas as pd, numpy as np, config

def calculate():
  metadata_df = get_dataframe()

  # Calculate correlation between all columns
  correlations = metadata_df.corr()

  # Store correlation matrix as CSV file
  correlations.to_csv(config.CORRELATION_LOCATION + '/leaveoneout_metadata_correlation_matrix.csv')


def get_dataframe():
  review_meta = pd.read_csv(config.TEXT_DATA_LOCATION + '/review metadata.csv')
  
  # Cut off columns not used for correlation calculation
  review_meta = review_meta.loc[:, 'is update':]

  # Get the leave one out results
  outcome_fetcher = fetcher.Outcome_fetcher()
  leaveoneout_matrix = outcome_fetcher.get_data('leaveoneout')

  # Cut off the identifier column and change data to float
  leaveoneout_matrix = leaveoneout_matrix[:, 1:].astype(float)

  # Get review mean performance
  leaveoneout_means = np.mean(leaveoneout_matrix, axis = 0)
  
  # Add performance into metadata matrix
  review_meta['performance'] = leaveoneout_means.transpose()

  review_meta['include_exclude_similarity'] = calculate_include_exclude_similarity().transpose()

  return review_meta


def calculate_include_exclude_similarity():
  tfidf_matrix = file_handle.load_matrix('tfidf_matrix')

  conn = db.Connector()
  review_indices = conn.get_review_indices()
  include_labels = conn.get_labels()
  
  cosine_vector = []

  # Loop over the review indices
  for l, r in review_indices:
    # Get the subset of review documents
    review_subset = tfidf_matrix[l:r, :]
    # Get the includes/excludes boolean array
    # for this review
    review_includes = np.array(include_labels[l:r])

    # Get the vectors of includes and excludes
    includes_vect = review_subset[review_includes, :].mean(axis = 0)
    excludes_vect = review_subset[~review_includes, :].mean(axis = 0)

    # Stack the vectors into a matrix
    cos_matrix = np.vstack([includes_vect, excludes_vect])

    # Calculate the similarity between includes
    # and excludes.
    cos_sim = cosine_similarity(cos_matrix)[0, 1]

    cosine_vector.append(cos_sim)

  # Turn list into numpy array for further processing
  return np.array(cosine_vector)
