import config

import itertools
import math
from nltk.tokenize import WhitespaceTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

tokenizer = WhitespaceTokenizer()

def list_merge(items):
  # Flatten tokenized documents back into a list of strings
  # Input: [['w1', 'w2'], ['w3', 'w4'], ...]
  # Output: [['w1 w2'], ['w3 w4'], ...]
  merged = list(map(lambda x: ' '.join(x), items))

  return merged

def merge_documents_as_string(items):
  merged = list_merge(items)

  return merged

def merge_title_abstract(titles, abstracts):
  # Zip titles and abstracts into tuples
  # Output: [(['title_w1', 'title_w2'], ['abstract_w1', 'abstract_w2']), ([...], [...])]
  zipped = zip(titles, abstracts)

  # Merge the tuples and turn them into lists
  # Output: [['title_w1', 'title_w2', 'abstract_w1', 'abstract_w2'], [...]]
  merged = [list(itertools.chain(*document)) for document in zipped]
  
  return merged

def merge_documents(document_list):
  # Starts as tokenized documents: [['w1', 'w2'], ['w3', 'w4'], ...]
  # Ends with flattened list of tokens without document boundaries: ['w1', 'w2', 'w3', 'w4', ...]
  merged_documents = [item for sublist in document_list for item in sublist]

  return merged_documents

def merge_all(titles, abstracts):
  titles = merge_documents(titles)
  abstracts = merge_documents(abstracts)

  return titles, abstracts

def merge_full(titles, abstracts):
  merged_titles = merge_documents_as_string(titles)
  merged_abstracts = merge_documents_as_string(abstracts)

  return merged_titles + merged_abstracts

def create_vectorizer(vectorize_function = CountVectorizer):
  vectorizer = vectorize_function(min_df = config.MIN_DOCFREQ, max_df = config.MAX_DOCFREQ)

  return vectorizer

def tokenize_item(item):
  return tokenizer.tokenize(item)

def tokenize_all(titles, abstracts):
  titles = list(map(tokenize_item, titles))
  abstracts = list(map(tokenize_item, abstracts))

  return titles, abstracts