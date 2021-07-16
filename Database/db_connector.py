import Database.db as db
from peewee import *

class Connector:

  def get_articles(self):
    return db.Article.select(db.Article.title, db.Article.abstract).execute()


  def get_labels(self):
    labels = db.Article.select(db.Article.included).execute()

    return [x.included for x in labels]


  def get_review_names(self):
    review_id_query = db.Article.select(db.Article.review_id).distinct().execute()

    review_ids = [x.review_id for x in review_id_query]

    return review_ids


  def get_review_indices(self):
    review_id_query = db.Article.select(db.Article.review_id).execute()

    review_ids = [x.review_id for x in review_id_query]

    indices = []

    last_review = None
    first_spotted = 0

    for i in range(len(review_ids)):
      review_id = review_ids[i]

      if last_review is None:
        last_review = review_id

      if i < len(review_ids) - 1:
        next_review_id = review_ids[i + 1]
        
        if next_review_id != last_review:
          indices.append((first_spotted, i))

          first_spotted = i + 1
          last_review = next_review_id
      else:
        indices.append((first_spotted, i))

    return indices
