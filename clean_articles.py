import Preprocessing.remove_characters as string_cleaning
import Preprocessing.tokenize as tokenize
import Preprocessing.remove_stopwords as stopwords
import Preprocessing.token_length as token_length
import Libs.file_storage as file_handle
import config
import pickle

import Text_data.test_file as test_data

import Database.db_connector as db

def get_corpus():
  if config.CLEAN_TEST_DATA is True:
    return test_data.titles, test_data.abstracts

  conn = db.Connector()

  articles = conn.get_articles()

  titles = []
  abstracts = []

  for article in articles:
    titles.append(article.title)
    abstracts.append(article.abstract)

  return titles, abstracts


def clean_corpus(titles, abstracts):
  cleaned_titles, cleaned_abstracts = string_cleaning.remove_all(titles, abstracts)
  print('cleaned')

  tokenized_titles = list(map(tokenize.tokenize_item, cleaned_titles))
  tokenized_abstracts = list(map(tokenize.tokenize_item, cleaned_abstracts))
  print('tokenized')

  cleaned_titles, cleaned_abstracts = stopwords.remove_all(tokenized_titles, tokenized_abstracts)
  print('stopwords removed')

  cleaned_titles, cleaned_abstracts = token_length.remove_all(cleaned_titles, cleaned_abstracts)
  print('length removed')

  return cleaned_titles, cleaned_abstracts


def run_corpus():
  print('fetching')
  raw_titles, raw_abstracts = get_corpus()

  print('cleaning')
  titles, abstracts = clean_corpus(raw_titles, raw_abstracts)

  # Merge the title and abstract lists into one
  print('merging title+abstract')
  corpus = tokenize.merge_title_abstract(titles, abstracts)
  # Merge the documents into strings, yields a list of strings
  print('merging documents')
  corpus = tokenize.merge_documents_as_string(corpus)

  # Store the merged corpus
  print('storing')
  file_handle.store_full(corpus)


if __name__ == "__main__":
  run_corpus()