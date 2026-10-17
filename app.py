
from flask import Flask, jsonify, request, send_from_directory
from movie_recommender import (
    movies, GENRES, similarity_df, find_movie, genres_of
)
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

app = Flask(__name__)
@app.route("/")
def home():
    return send_from_directory(".", "index.html")

    return """
    <h1>🎬 CineMatch AI</h1>
    <p>Your Movie Recommendation API is running!</p>
    <p>Try these links:</p>
    <ul>
        <li><a href="/api/movies">View all movies</a></li>
        <li><a href="/api/recommend?movie=Inception">Recommend movies similar to Inception</a></li>
    </ul>
    """

@app.get("/api/movies")
def get_movies():
    return jsonify({
        "movies": movies["Movie"].tolist(),
        "genres": GENRES
    })

@app.get("/api/recommend")
def recommend_movie():
    name = request.args.get("movie", "")
    match = find_movie(name)

    if not match:
        return jsonify({
            "error": "Movie not found",
            "available": movies["Movie"].tolist()
        }), 404

    scores = (
        similarity_df[match]
        .drop(match)
        .sort_values(ascending=False)
        .head(3)
    )

    results = [
        {
            "title": title,
            "genres": genres_of(title),
            "score": float(score)
        }
        for title, score in scores.items()
    ]

    return jsonify({
        "movie": match,
        "genres": genres_of(match),
        "recommendations": results
    })

@app.get("/api/genres")
def recommend_genres():
    raw = request.args.get("genres", "")
    selected = {
        item.strip().lower()
        for item in raw.split(",")
        if item.strip()
    }

    valid = [
        genre for genre in GENRES
        if genre.lower() in selected
    ]

    if not valid:
        return jsonify({
            "error": "Select at least one valid genre",
            "available_genres": GENRES
        }), 400

    profile = np.array([
        [1 if genre in valid else 0 for genre in GENRES]
    ])

    scores = cosine_similarity(
        profile,
        movies[GENRES]
    )[0]

    ranked = sorted(
        zip(movies["Movie"], scores),
        key=lambda item: item[1],
        reverse=True
    )[:3]

    return jsonify({
        "genres": valid,
        "recommendations": [
            {
                "title": title,
                "genres": genres_of(title),
                "score": float(score)
            }
            for title, score in ranked
        ]
    })

if __name__ == "__main__":
    app.run(debug=True)
