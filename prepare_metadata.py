import Libs.file_storage as corpus_handle
import pickle, config
from itertools import groupby

def get_word_metadata(matrix, unique):
  if unique is True:
    matrix = matrix > 0
    matrix = matrix.astype(int)

  row_sums = matrix.sum(axis = 1)

  return {'min': row_sums.min(), 'max': row_sums.max(), 'mean': row_sums.mean()}


def do_analysis():
  initial_corpus = corpus_handle.load_matrix('tfidf_matrix')

  counts = {}

  counts['counts'] = get_word_metadata(initial_corpus, False)
  counts['unique'] = get_word_metadata(initial_corpus, True)

  with open(config.TEXT_DATA_LOCATION + '/word_counts.pickle', 'wb') as handle:
    print(counts)
    pickle.dump(counts, handle, protocol = pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
  do_analysis()
