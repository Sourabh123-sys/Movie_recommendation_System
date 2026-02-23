from flask import Flask, render_template, request
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load data
data = pd.read_pickle("data.pkl")
matrix = pickle.load(open("matrix.pkl", "rb"))

# Get movie list for dropdown
movie_list = sorted(data['title'].dropna().unique())

def recommend(title):
    movie_data = data[data['title'] == title]

    if movie_data.empty:
        return ["Movie not found"]

    idx = movie_data.index[0]
    scores = cosine_similarity(matrix[idx], matrix).flatten()
    top_indices = scores.argsort()[-6:-1][::-1]

    return list(data.iloc[top_indices]['title'])

@app.route("/")
def home():
    return render_template("index.html", movies=movie_list)

@app.route("/recommend", methods=["POST"])
def get_recommendations():
    selected_movie = request.form["movie"]
    recommendations = recommend(selected_movie)
    return render_template("index.html", movies=movie_list,
                           selected_movie=selected_movie,
                           recommendations=recommendations)

if __name__ == "__main__":
    app.run(debug=True)
