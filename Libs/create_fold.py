import Database.db_connector as db
import Libs.file_storage as file_handle
import config, math, numpy as np

class Create_fold(object):
  conn = None
  num_reviews = None
  review_indices = None
  review_names = None
  y = None
  X = None
  x_test_total_size = 0
  experiment_type = ''
  split_reviews = []
  cosine_similarity = None
  cut_review_subset = None


  def __init__(self, X, experiment_type):
    self.conn = db.Connector()
    # Retrieve a list of tuples with the starting and ending
    # indices for each review in the dataset.
    self.review_indices = self.conn.get_review_indices()
    self.review_names = self.conn.get_review_names()
    # Retrieve the article labels
    self.y = np.asarray(self.conn.get_labels())
    self.X = X

    self.num_reviews = self.get_num_reviews()

    # Cut the review matrix to the number of reviews
    # defined in the config (MAX_REVIEWS).
    self.subset_matrix()

    self.experiment_type = experiment_type

    if 'n_vs_one' in self.experiment_type:
      self.cosine_similarity = file_handle.load_matrix('cosine_sim_matrix')


  def determine_folds(self):
    if self.cut_review_subset is not None:
      return len(self.cut_review_subset)

    test_size = config.NUM_REVIEWS_TEST_SET

    # Rudimental, if you want to cover the whole dataset set
    # the NUM_REVIEWS_TEST_SET so that it divides the full
    # dataset in equal parts. For example, there are 50 reviews
    # set the NUM_REVIEWS_TEST_SET to <1, 2, 5, 10, etc.>
    folds = math.floor(self.get_num_reviews() / test_size)
    
    if config.MAX_FOLDS is not -1 and config.MAX_FOLDS < folds:
      folds = config.MAX_FOLDS + 1

    return folds


  def get_num_reviews(self):
    # Determine the fold step size
    num_reviews = len(self.review_indices)

    if config.MAX_REVIEWS is not -1 and config.MAX_REVIEWS < num_reviews:
      num_reviews = config.MAX_REVIEWS

    return num_reviews


  def subset_matrix(self):
    # Determine the last document index that is included.
    last_review = self.review_indices[self.num_reviews - 1]
    last_index = last_review[1]

    # Create a mask to cut the X and y inputs.
    mask = np.zeros(self.X.shape[0], dtype = bool)
    mask[range(0, last_index + 1)] = True
    
    self.X = self.X[mask,:]
    self.y = self.y[mask]


  def get_fold(self, fold):
    if self.experiment_type == 'leave_one_out':
      return self.get_leaveoneout(fold)

    if self.experiment_type == 'one_vs_one':
      return self.get_onevsone(fold)

    if self.experiment_type == 'n_vs_one':
      return self.get_nvsone(fold)

    if self.experiment_type == 'n_vs_one_random':
      return self.get_nvsone_random(fold)


  def get_leaveoneout(self, fold):
    # Get the index of the first and last review for this fold
    first_test_review = (fold - 1) * config.NUM_REVIEWS_TEST_SET
    last_test_review = first_test_review - 1 + config.NUM_REVIEWS_TEST_SET

    if first_test_review != last_test_review:
      selected_review_names = self.review_names[first_test_review:last_test_review]
    else:
      selected_review_names = [self.review_names[first_test_review]]
    
    # Determine the minimum document index and maximum
    # document index.
    min_review_index = self.review_indices[first_test_review][0]
    max_review_index = self.review_indices[last_test_review][1]
    
    # Get individual review indices
    test_review_indices = []

    for x in self.review_indices[first_test_review:last_test_review + 1]:
      test_review_indices.append((x[0] - min_review_index, (x[1] - min_review_index) + 1))

    mask = np.ones(self.X.shape[0], dtype = bool)
    mask[range(min_review_index, max_review_index + 1)] = False
    
    # Slice matrices into train and test sets
    X_train = self.X[mask,:]
    X_test = self.X[~mask,:]
    
    self.x_test_total_size = self.x_test_total_size + X_test.shape[0]
    
    y_train = self.y[mask]
    y_test = self.y[~mask]
    
    obj = {'X_train': X_train, 'y_train': y_train, 'X_test': X_test, 'y_test': y_test, 'review_indices': test_review_indices, 'test_review_names': selected_review_names}

    return obj


  def get_onevsone(self, fold):
    # Get the index of the first and last review for this fold
    train_review = (fold - 1)

    selected_review_names = self.review_names
    
    # Determine the minimum document index and maximum
    # document index.
    min_review_index = self.review_indices[train_review][0]
    max_review_index = self.review_indices[train_review][1]
    
    # Get individual review indices
    test_review_indices = self.review_indices
    
    mask = np.ones(self.X.shape[0], dtype = bool)
    mask[range(min_review_index, max_review_index + 1)] = False
    
    # Slice matrices into train and test sets
    X_train = self.X[~mask,:]
    X_test = self.X
    
    y_train = self.y[~mask]
    y_test = self.y
    
    self.x_test_total_size = self.x_test_total_size + X_train.shape[0]
    
    obj = {'X_train': X_train, 'y_train': y_train, 'X_test': X_test, 'y_test': y_test, 'review_indices': test_review_indices, 'test_review_names': selected_review_names}

    return obj


  def get_nvsone(self, fold):
    # Get the index of the first and last review for this fold
    test_review = (fold - 1)

    # Get the cosine similarities ordered and remove the 
    # review itself.
    similarities = self.cosine_similarity.iloc[[test_review]]
    test_review_name = similarities.columns[test_review]

    similarities = similarities.drop(test_review_name, axis = 1)
    similarities = similarities.sort_values(by = test_review_name, ascending = False, axis = 1)

    # Find the indexes of the reviews and reorder the
    # review indices array.
    reorder = [self.review_names.index(i) for i in similarities.columns]
    train_review_indices = [self.review_indices[i] for i in reorder]

    selected_review_names = similarities.columns
    
    test_review_indices = self.review_indices[test_review]

    # Determine the test review indices
    min_review_index = test_review_indices[0]
    max_review_index = test_review_indices[1]

    # Subset the feature matrix
    mask = np.ones(self.X.shape[0], dtype = bool)
    mask[range(min_review_index, max_review_index + 1)] = False
    
    X_test = self.X[~mask,:]
    y_test = self.y[~mask]

    # Determine new indices for X_test set
    shift_test_review_indices = [(0, (max_review_index - min_review_index) + 1)]
    
    self.x_test_total_size = self.x_test_total_size + X_test.shape[0]

    # Create training sets for each N
    X_train_sets = []
    y_train_sets = []
    set_train_names = []
    
    for size in config.SIMILARITY_STEPS:
      selected_reviews = train_review_indices[0:size]
      
      mask = np.ones(self.X.shape[0], dtype = bool)

      for review in selected_reviews:
        mask[range(review[0], review[1] + 1)] = False

      X_subset = self.X[~mask,:]
      y_subset = self.y[~mask]

      X_train_sets.append(X_subset)
      y_train_sets.append(y_subset)

      set_train_names.append(selected_review_names[0:size].tolist())
    
    obj = {'X_train': X_train_sets, 'y_train': y_train_sets, 'X_test': X_test, 'y_test': y_test, 'review_indices': shift_test_review_indices, 'test_review_names': [test_review_name], 'train_review_names': set_train_names}

    return obj


  def get_nvsone_random(self, fold):
    # Get the index of the first and last review for this fold
    test_review = (fold - 1)

    # Get the cosine similarities ordered and remove the 
    # review itself.
    similarities = self.cosine_similarity.iloc[[test_review]]
    test_review_name = similarities.columns[test_review]

    similarities = similarities.drop(test_review_name, axis = 1)

    random_order = np.array(similarities.columns)

    np.random.shuffle(random_order)

    similarities = similarities.loc[:, random_order]
    
    # Find the indexes of the reviews and reorder the
    # review indices array.
    reorder = [self.review_names.index(i) for i in similarities.columns]

    train_review_indices = [self.review_indices[i] for i in reorder]

    selected_review_names = similarities.columns
    
    test_review_indices = self.review_indices[test_review]

    # Determine the test review indices
    min_review_index = test_review_indices[0]
    max_review_index = test_review_indices[1]

    # Subset the feature matrix
    mask = np.ones(self.X.shape[0], dtype = bool)
    mask[range(min_review_index, max_review_index + 1)] = False
    
    X_test = self.X[~mask,:]
    y_test = self.y[~mask]

    # Determine new indices for X_test set
    shift_test_review_indices = [(0, (max_review_index - min_review_index) + 1)]
    
    self.x_test_total_size = self.x_test_total_size + X_test.shape[0]

    # Create training sets for each N
    X_train_sets = []
    y_train_sets = []
    set_train_names = []
    
    for size in config.SIMILARITY_STEPS:
      selected_reviews = train_review_indices[0:size]
      
      mask = np.ones(self.X.shape[0], dtype = bool)

      for review in selected_reviews:
        mask[range(review[0], review[1] + 1)] = False

      X_subset = self.X[~mask,:]
      y_subset = self.y[~mask]

      X_train_sets.append(X_subset)
      y_train_sets.append(y_subset)

      set_train_names.append(selected_review_names[0:size].tolist())
    
    obj = {'X_train': X_train_sets, 'y_train': y_train_sets, 'X_test': X_test, 'y_test': y_test, 'review_indices': shift_test_review_indices, 'test_review_names': [test_review_name], 'train_review_names': set_train_names}

    return obj
