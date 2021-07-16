import Libs.file_storage as file_handle
import Libs.outcome_fetcher as fetcher
import pandas as pd, numpy as np, config

def calculate():
  performance_df = get_data()
  similarity_df = get_average_similarity()

  # Merge the two dataframes column-wise
  df = pd.concat([performance_df, similarity_df], axis = 1)

  # Calculate correlation between all columns
  correlations = df.corr()

  # Store correlation matrix as CSV file
  correlations.to_csv(config.CORRELATION_LOCATION + '/nvsone_performance_to_similarity_correlation_matrix.csv')


def get_data():
  # Get the n vs one results
  outcome_fetcher = fetcher.Outcome_fetcher()
  nvsone_matrix, _, _, _ = outcome_fetcher.get_data('nvsone')

  nvsone_labels = config.SIMILARITY_STEPS
  num_runs = int(nvsone_matrix.shape[0] / len(nvsone_labels))
  
  step_mean_values = None

  # Loop over the similarity steps
  for i in range(0, len(nvsone_labels)):
    nvsone_sim_step_values = []

    # Loop over the runs for each similarity step
    for run_num in range(0, num_runs):
      nvsone_sim_step_values.append(nvsone_matrix[i * run_num, 1:].tolist())

    # Get the mean performance values for each
    # review in this similarity step (i.e. training size)
    mean_values = np.array(np.mean(nvsone_sim_step_values, axis = 0))
    
    # Build the matrix
    if step_mean_values is None:
      step_mean_values = mean_values
    else:
      step_mean_values = np.vstack([step_mean_values, mean_values])

  # Transpose matrix to match with the similarity matrix
  step_mean_values = step_mean_values.transpose()

  # Create labels for the pandas dataframe
  labels = ['performance for size %i' % i for i in config.SIMILARITY_STEPS]

  # Return as a pandas dataframe
  return pd.DataFrame(step_mean_values, columns = labels)


def get_average_similarity():
  cosine_similarity = file_handle.load_matrix('cosine_sim_matrix')

  similarity_matrix = None

  # Loop over the columns of the similarity matrix
  for col in cosine_similarity.columns:
    # Get a single review,
    # remove the review itself,
    # and sort the similarity values
    cut_similarities = cosine_similarity.loc[[col]]
    cut_similarities = cut_similarities.drop(col, axis = 1)
    cut_similarities = cut_similarities.sort_values(by = col, ascending = False, axis = 1)

    size_similarity = []

    # Loop over the similarity steps (i.e. training sizes)
    for size in config.SIMILARITY_STEPS:
      # Fetch the subset of similarities for
      # this similarity step
      size_subset = cut_similarities.iloc[:, 0:size]

      # Get the mean similarity value for this subset
      size_similarity.append(size_subset.mean(axis = 1))

    # Build an array of mean similarity values
    # transpose to match with the performance matrix
    size_similarity = np.array(size_similarity).transpose()

    # Build the matrix
    if similarity_matrix is None:
      similarity_matrix = size_similarity
    else:
      similarity_matrix = np.vstack([similarity_matrix, size_similarity])

  # Create labels for the pandas dataframe
  labels = ['similarity for size %i' % i for i in config.SIMILARITY_STEPS]

  # Return as a pandas dataframe
  return pd.DataFrame(similarity_matrix, columns = labels)
