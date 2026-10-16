"""
=====================================================================
  CineMatch AI - Content-Based Movie Recommendation System
=====================================================================
  Technique : Content-Based Filtering using Cosine Similarity
  Libraries : pandas, numpy, scikit-learn, matplotlib

  How it works 
  1. Every movie is converted into a numeric "genre vector"
     (1 = has the genre, 0 = does not).
  2. Cosine similarity measures the angle between two vectors.
     Score 1.0 = identical taste, 0.0 = nothing in common.
  3. For a chosen movie we rank all other movies by similarity.
  4. Bonus: we can also build a USER PROFILE from favourite genres
     and recommend movies that match the profile.
=====================================================================
"""

import difflib
import os

import matplotlib
matplotlib.use("Agg")          # saves images without needing a display window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------
# 1. DATASET  (1 = movie has this genre, 0 = does not)
# ---------------------------------------------------------------------
GENRES = ["Action", "SciFi", "Romance", "Horror", "Comedy", "Thriller", "Drama"]

MOVIE_DATA = {
    #  Movie            Action SciFi Romance Horror Comedy Thriller Drama
    "Avengers":        [1, 1, 0, 0, 0, 0, 0],
    "Iron Man":        [1, 1, 0, 0, 0, 0, 0],
    "Inception":       [1, 1, 0, 0, 0, 1, 0],
    "Interstellar":    [0, 1, 0, 0, 0, 0, 1],
    "The Matrix":      [1, 1, 0, 0, 0, 1, 0],
    "Titanic":         [0, 0, 1, 0, 0, 0, 1],
    "The Notebook":    [0, 0, 1, 0, 0, 0, 1],
    "La La Land":      [0, 0, 1, 0, 1, 0, 1],
    "The Conjuring":   [0, 0, 0, 1, 0, 1, 0],
    "Annabelle":       [0, 0, 0, 1, 0, 0, 0],
    "Get Out":         [0, 0, 0, 1, 0, 1, 0],
    "Hangover":        [0, 0, 0, 0, 1, 0, 0],
    "Superbad":        [0, 0, 0, 0, 1, 0, 0],
    "Shutter Island":  [0, 0, 0, 0, 0, 1, 1],
    "John Wick":       [1, 0, 0, 0, 0, 1, 0],
}

movies = pd.DataFrame.from_dict(MOVIE_DATA, orient="index", columns=GENRES)
movies.index.name = "Movie"
movies = movies.reset_index()
features = movies[GENRES]

# ---------------------------------------------------------------------
# 2. SIMILARITY MATRIX  (every movie vs every other movie)
# ---------------------------------------------------------------------
similarity = cosine_similarity(features)
similarity_df = pd.DataFrame(similarity, index=movies["Movie"], columns=movies["Movie"])


# ---------------------------------------------------------------------
# 3. HELPER FUNCTIONS
# ---------------------------------------------------------------------
def find_movie(name):
    """Case-insensitive lookup with 'did you mean' spell-correction."""
    lookup = {m.lower(): m for m in movies["Movie"]}
    key = name.strip().lower()
    if key in lookup:
        return lookup[key]
    close = difflib.get_close_matches(key, lookup.keys(), n=1, cutoff=0.6)
    return lookup[close[0]] if close else None


def genres_of(movie_name):
    row = movies[movies["Movie"] == movie_name][GENRES].iloc[0]
    return ", ".join(g for g in GENRES if row[g] == 1)


def bar(score, width=20):
    filled = int(round(score * width))
    return "#" * filled + "-" * (width - filled)


def recommend(movie_name, n=3):
    """Recommend n movies similar to movie_name. Returns a DataFrame."""
    match = find_movie(movie_name)
    if match is None:
        print(f"\n  Sorry, '{movie_name}' is not in the database.")
        print("  Available movies:", ", ".join(movies["Movie"]), "\n")
        return None

    scores = similarity_df[match].drop(match).sort_values(ascending=False).head(n)

    print(f"\n  If you liked '{match}'  [{genres_of(match)}]")
    print("  you may also like:")
    print("  " + "-" * 62)
    for rank, (title, score) in enumerate(scores.items(), start=1):
        print(f"  {rank}. {title:<15} {bar(score)} {score * 100:5.1f}% match")
        print(f"     Genres: {genres_of(title)}")
    print()
    return scores.reset_index().rename(columns={match: "Similarity"})


def recommend_by_genres(liked_genres, n=3):
    """Build a user-profile vector from favourite genres and rank movies."""
    valid = [g for g in GENRES if g.lower() in [x.lower() for x in liked_genres]]
    if not valid:
        print("\n  No valid genres entered. Choose from:", ", ".join(GENRES), "\n")
        return None

    profile = np.array([[1 if g in valid else 0 for g in GENRES]])
    scores = cosine_similarity(profile, features)[0]
    result = (pd.DataFrame({"Movie": movies["Movie"], "Similarity": scores})
              .sort_values("Similarity", ascending=False).head(n))

    print(f"\n  Your taste profile: {', '.join(valid)}")
    print("  Top picks for you:")
    print("  " + "-" * 62)
    for rank, row in enumerate(result.itertuples(index=False), start=1):
        print(f"  {rank}. {row.Movie:<15} {bar(row.Similarity)} {row.Similarity * 100:5.1f}% match")
    print()
    return result


def save_heatmap(path="similarity_heatmap.png"):
    """Visualise the similarity matrix as a heatmap image."""
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(similarity, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(movies)))
    ax.set_yticks(range(len(movies)))
    ax.set_xticklabels(movies["Movie"], rotation=60, ha="right")
    ax.set_yticklabels(movies["Movie"])
    fig.colorbar(im, ax=ax, label="Cosine similarity")
    ax.set_title("CineMatch AI - Movie Similarity Heatmap")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Heatmap saved as: {os.path.abspath(path)}\n")


# ---------------------------------------------------------------------
# 4. DEMO + INTERACTIVE MENU
# ---------------------------------------------------------------------
BANNER = r"""
  ============================================================
     CineMatch AI  -  Movie Recommendation System
     Content-Based Filtering | Cosine Similarity
  ============================================================
"""


def demo():
    print("  --- AUTOMATIC DEMO ---")
    recommend("Inception")
    recommend("Titanic", 1)
    recommend("The Conjuring", 2)
    recommend_by_genres(["Comedy", "Romance"], 3)


def menu():
    while True:
        print("  MENU")
        print("   1. Recommend movies similar to a movie I like")
        print("   2. Recommend movies from my favourite genres")
        print("   3. Show all movies in the database")
        print("   4. Save similarity heatmap (picture)")
        print("   5. Exit")
        choice = input("\n  Enter your choice (1-5): ").strip()

        if choice == "1":
            name = input("  Enter a movie name: ")
            recommend(name, 3)
        elif choice == "2":
            print("  Genres:", ", ".join(GENRES))
            raw = input("  Enter favourite genres (comma separated): ")
            recommend_by_genres([g.strip() for g in raw.split(",")], 3)
        elif choice == "3":
            print()
            for _, row in movies.iterrows():
                print(f"   - {row['Movie']:<15} [{genres_of(row['Movie'])}]")
            print()
        elif choice == "4":
            save_heatmap()
        elif choice == "5":
            print("\n  Thank you for using CineMatch AI!\n")
            break
        else:
            print("\n  Invalid choice. Please enter a number from 1 to 5.\n")


if __name__ == "__main__":
    print(BANNER)
    demo()
    try:
        save_heatmap()
        menu()
    except (EOFError, KeyboardInterrupt):
        print("\n\n  Program closed. Goodbye!\n")
