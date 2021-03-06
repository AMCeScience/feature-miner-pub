import config
import datetime
from peewee import *

conn = SqliteDatabase(config.DB_FILE)

class BaseModel(Model):
  class Meta:
    database = conn

class Journal(BaseModel):
  title = CharField(max_length = 150)
  iso = CharField(max_length = 100)
  iso_stripped = CharField(max_length = 100)
  issn = CharField(max_length = 45, null = True)
  created_date = DateTimeField(default = datetime.datetime.now)

class Article(BaseModel):
  pubmed_id = CharField(max_length = 45)
  title = CharField(max_length = 300)
  title_stripped = CharField(max_length = 300)
  abstract = TextField(null = True)
  journal = ForeignKeyField(Journal, null = True)
  publication_date = DateField(null = True)
  doi = CharField(max_length = 45, null = True)
  created_date = DateTimeField(default = datetime.datetime.now)
  included = BooleanField()
  review_id = CharField(max_length = 10)

class Keyword(BaseModel):
  keyword = CharField(max_length = 45)
  created_date = DateTimeField(default = datetime.datetime.now)

class Keyword_to_article(BaseModel):
  article = ForeignKeyField(Article)
  keyword = ForeignKeyField(Keyword)
  created_date = DateTimeField(default = datetime.datetime.now)

  class Meta:
    index = (
      (('article', 'keyword'), True),
    )

class Search_results(BaseModel):
  pubmed_id = CharField(max_length = 45)
  search_query = CharField(max_length = 200)
  fetched = BooleanField(default = False)
  created_date = DateTimeField(default = datetime.datetime.now)

# Instantiate the database
if not Article.table_exists():
  conn.create_tables([Journal, Article, Keyword, Keyword_to_article, Search_results])
