import pandas as pd
import pickle
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

print("Loading datasets...")

movies = pd.read_csv("data/movies_metadata.csv", low_memory=False)
credits = pd.read_csv("data/credits.csv")
keywords = pd.read_csv("data/keywords.csv")
links_small = pd.read_csv("data/links_small.csv")

# Basic cleaning
movies = movies[['id','title','overview','genres','vote_average']]
movies.dropna(inplace=True)

movies['id'] = movies['id'].astype(str)
credits['id'] = credits['id'].astype(str)
keywords['id'] = keywords['id'].astype(str)

# Merge
data = movies.merge(credits, on='id')
data = data.merge(keywords, on='id')

# Use only small dataset movies (RAM safe)
links_small = links_small[links_small['tmdbId'].notnull()]
links_small['tmdbId'] = links_small['tmdbId'].astype(int).astype(str)

data = data[data['id'].isin(links_small['tmdbId'])]
data = data.reset_index(drop=True)


print("Movies used:", len(data))

# Clean genres
def clean_genres(text):
    return " ".join([i['name'] for i in ast.literal_eval(text)])

# Clean keywords
def clean_keywords(text):
    return " ".join([i['name'] for i in ast.literal_eval(text)])

# Clean cast (top 3)
def clean_cast(text):
    return " ".join([i['name'] for i in ast.literal_eval(text)[:3]])

data['genres'] = data['genres'].apply(clean_genres)
data['keywords'] = data['keywords'].apply(clean_keywords)
data['cast'] = data['cast'].apply(clean_cast)

# Combined text
data['combined'] = (
    data['overview'] + " " +
    data['genres'] + " " +
    data['keywords'] + " " +
    data['cast']
)

# ---------------------------
# Content TF-IDF (RAM SAFE)
# ---------------------------

tfidf_rec = TfidfVectorizer(
    stop_words='english',
    max_features=5000
)

matrix = tfidf_rec.fit_transform(data['combined'])

pickle.dump(data, open("data.pkl", "wb"))
pickle.dump(matrix, open("matrix.pkl", "wb"))
pickle.dump(tfidf_rec, open("tfidf_rec.pkl", "wb"))

print("Content model saved.")

# ---------------------------
# Classification Model
# ---------------------------

data['label'] = data['vote_average'].apply(
    lambda x: 1 if float(x) > 7 else 0
)

tfidf_cls = TfidfVectorizer(
    stop_words='english',
    max_features=3000
)

X = tfidf_cls.fit_transform(data['overview'])
y = data['label']

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(tfidf_cls, open("tfidf_cls.pkl", "wb"))

print("Classification model saved.")

print("🔥 Hybrid RAM-Optimized System Ready!")
