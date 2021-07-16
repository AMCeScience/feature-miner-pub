from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import Database.db_connector as db
import Libs.file_storage as file_handle
import Preprocessing.tokenize as tokenize
import config, random, pickle, itertools, numpy as np, pandas as pd, time
from multiprocessing import Pool
from functools import partial

def get_articles():
  return file_handle.load_full()


def create_features():
  # Create a normalised term frequency matrix
  print('get TF-IDF matrix')
  tfidf_matrix = get_create_matrix()

  # Create a cosine similarity matrix for every review
  print('get similarity matrix')
  similarity_matrix = get_cosine_similarity(tfidf_matrix)

  # Create five topicmodels to account for randomisation.
  # Fix parameters
  parallel_func = partial(run_single_topicmodel)

  with Pool() as pool:
    # Parallelise
    pool.map(parallel_func, range(1, config.NUM_TOPICMODELS + 1))


def run_single_topicmodel(i):
  print('get TM matrix #%i' % i)
  tm_matrix = get_create_topicmodels(i)


def get_create_topicmodels(i):
  existing_tm = file_handle.load_topicmodel(i)

  if existing_tm is not None:
    return existing_tm['matrix']

  # Retrieve the articles
  X = get_articles()

  # Get a CountVectorizer and apply it to the articles
  tf_vectorizer = tokenize.create_vectorizer(CountVectorizer)
  tf_matrix = tf_vectorizer.fit_transform(X)

  seed = random.randint(0, 2**32 - 1)

  # Get a LDA object and fit to the matrix
  print('get LDA object')
  tm_obj = LatentDirichletAllocation(
    n_components = 200,
    learning_method = 'online',
    max_iter = 5,
    random_state = seed)
  tm_obj.fit(tf_matrix)
  # transform function returns a document-topic distribution matrix
  tm_matrix = tm_obj.transform(tf_matrix)

  # Store the topicmodel in a pickle file
  print('storing TM matrix')
  file_handle.store_topicmodel(i, {'matrix': tm_matrix, 'seed': seed})

  return tm_matrix


def get_create_matrix():
  existing_corpus = file_handle.load_matrix('tfidf_matrix')

  if existing_corpus is not None:
    return existing_corpus

  # Retrieve the articles
  X = get_articles()

  # Get a CountVectorizer and apply it to the articles
  tfidf_vectorizer = tokenize.create_vectorizer(TfidfVectorizer)
  tfidf_matrix = tfidf_vectorizer.fit_transform(X)

  # Scaling the data
  # https://neerajkumar.org/writings/svm/
  print('scaling TF-IDF matrix')
  scaler = StandardScaler(with_mean = False)
  norm_tfidf_matrix = scaler.fit_transform(tfidf_matrix)

  # Store the matrix in a pickle file
  print('storing TF-IDF matrix')
  file_handle.store_matrix('tfidf_matrix', norm_tfidf_matrix)

  return norm_tfidf_matrix


def get_cosine_similarity(matrix):
  existing_similarity_matrix = file_handle.load_matrix('cosine_sim_matrix')
  existing_group_similarity_matrix = file_handle.load_matrix('group_sim_matrix')

  if existing_similarity_matrix is not None and existing_group_similarity_matrix is not None:
    return existing_similarity_matrix

  conn = db.Connector()
  review_indices = conn.get_review_indices()
  labels = conn.get_review_names()

  review_means = list()

  # Loop over the review indices
  for l,r in review_indices:
    # Get the subset of review documents
    review_subset = matrix[l:r,:]

    # Get the mean vector for this review
    review_means.append(np.mean(review_subset, axis = 0))

  # Stack the mean vectors of all reviews into a matrix
  review_matrix = np.vstack(review_means)

  # Calculate the cosine similarity between all reviews
  cos_sim = cosine_similarity(review_matrix)

  # Put into a pandas dataframe where we can add row and
  # column labels with the review label (e.g. CD010276)
  df = pd.DataFrame(cos_sim, index = labels, columns = labels)

  # Store as a pickle file and csv file
  file_handle.store_matrix('cosine_sim_matrix', df)
  df.to_csv(config.TEXT_DATA_LOCATION + '/cosine_sim_matrix.csv')

  get_group_similarity(labels, review_matrix)

  return cos_sim


def get_group_similarity(labels, mean_review_matrix):
  group_labels = [1, 2, 3, 4, 5, 6, 7, 'Other']

  group_matrix = None

  for _, group in config.GROUPS.items():
    idxs = [labels.index(i) for i in group]

    group_mean = mean_review_matrix[idxs, :].mean(axis = 0)

    if group_matrix is None:
      group_matrix = np.matrix(group_mean)
    else:
      group_matrix = np.vstack([group_matrix, group_mean])

  group_sim = cosine_similarity(group_matrix)

  df = pd.DataFrame(group_sim, index = group_labels, columns = group_labels)

  file_handle.store_matrix('group_sim_matrix', df)
  df.to_csv(config.TEXT_DATA_LOCATION + '/group_sim_matrix.csv')


if __name__ == "__main__":
  create_features()