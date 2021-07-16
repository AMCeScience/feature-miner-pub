import config, pickle

def store_classifier(i, feature_type, classifier_obj, dataset_num = None):
  dataset_name = 'classifier'

  if feature_type == 'tm':
    dataset_name = 'classifier_%i' % dataset_num

  with open(config.CLASSIFIER_LOCATION + '/%i_%s_%s.pickle' % (i, feature_type, dataset_name), 'wb') as handle:
    pickle.dump(classifier_obj, handle, protocol = pickle.HIGHEST_PROTOCOL)


def load_classifier(i, feature_type, dataset_num = None):
  dataset_name = 'classifier'

  if feature_type == 'tm':
    dataset_name = 'classifier_%i' % dataset_num

  filename = config.CLASSIFIER_LOCATION + '/%i_%s_%s.pickle' % (i, feature_type, dataset_name)

  if file_exists(filename):
    return read_file(filename)

  return None


def store_classifier_part(i, feature_type, part, classifier_obj):
  dataset_name = 'classifier'

  with open(config.CLASSIFIER_PART_LOCATION + '/%i_%s_%s_part%i.pickle' % (i, feature_type, dataset_name, part), 'wb') as handle:
    pickle.dump(classifier_obj, handle, protocol = pickle.HIGHEST_PROTOCOL)


def load_classifier_part(i, feature_type, part):
  dataset_name = 'classifier'

  filename = config.CLASSIFIER_PART_LOCATION + '/%i_%s_%s_part%i.pickle' % (i, feature_type, dataset_name, part)

  if file_exists(filename):
    return read_file(filename)

  return None


def load_external_classifier(experiment, num_run, i, feature_type, dataset_num = None):
  dataset_name = 'classifier'

  if dataset_num is not None:
    dataset_name = 'classifier_%i' % dataset_num

  filename = config.CLASSIFIER_LOCATION + '/%s/run_%i/%i_%s_%s.pickle' % (experiment, num_run, i, feature_type, dataset_name)

  if file_exists(filename):
    return read_file(filename)

  return None


def load_full():
  with open(config.TEXT_DATA_LOCATION + '/full_corpus.pickle', 'rb') as handle:
    corpus = pickle.load(handle)

  return corpus


def store_full(corpus):
  with open(config.TEXT_DATA_LOCATION + '/full_corpus.pickle', 'wb') as handle:
    pickle.dump(corpus, handle, protocol = pickle.HIGHEST_PROTOCOL)


def load_matrix(name):
  filename = config.TEXT_DATA_LOCATION + '/%s.pickle' % name

  if file_exists(filename):
    return read_file(filename)

  return None


def store_matrix(name, matrix_object):
  with open(config.TEXT_DATA_LOCATION + '/%s.pickle' % name, 'wb') as handle:
    pickle.dump(matrix_object, handle, protocol = pickle.HIGHEST_PROTOCOL)


def load_topicmodel(i):
  filename = config.TEXT_DATA_LOCATION + '/%i_topicmodel.pickle' % i

  if file_exists(filename):
    return read_file(filename)

  return None


def store_topicmodel(i, topicmodel):
  with open(config.TEXT_DATA_LOCATION + '/%i_topicmodel.pickle' % i, 'wb') as handle:
    pickle.dump(topicmodel, handle, protocol = pickle.HIGHEST_PROTOCOL)


def load_combination_set(i):
  filename = config.TEXT_DATA_LOCATION + '/combinations_%i.pickle' % i

  if file_exists(filename):
    return read_file(filename)

  return None


def store_combination_set(i, options):
  with open(config.TEXT_DATA_LOCATION + '/combinations_%i.pickle' % i, 'wb') as handle:
    pickle.dump(options, handle, protocol = pickle.HIGHEST_PROTOCOL)


def file_exists(name):
  try:
    with open(name, 'rb') as handle:
      return handle
  except (FileNotFoundError, OSError, IOError) as e:
    return False


def read_file(name):
  with open(name, 'rb') as handle:
    return pickle.load(handle)


def unpack_dataset(data):
  return (data['X_train'], data['y_train'], data['X_test'], data['y_test'], data['review_indices'], data['test_review_names'])