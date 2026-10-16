# CineMatch AI - Movie Recommendation System

A content-based movie recommender built with Python and Machine Learning.

## Concept
Each movie is a vector of genres (1 = yes, 0 = no). **Cosine similarity**
measures how close two vectors are, so movies with similar genres get a high
score and are recommended together.

## Features
- Recommend movies similar to a movie you like
- Recommend movies from your favourite genres (user-profile vector)
- Spell-correction ("incepton" -> Inception)
- Similarity heatmap image (`similarity_heatmap.png`)
- Interactive menu

## Setup
```
pip install -r requirements.txt
python movie_recommender.py
```

## Tech Stack
Python, pandas, NumPy, scikit-learn, Matplotlib

## Future Scope
Add ratings, TF-IDF on plot summaries, a larger dataset (MovieLens), a web UI (Streamlit).
