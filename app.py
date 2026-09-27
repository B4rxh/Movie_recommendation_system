import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

MOVIES = [
    {"title": "The Matrix", "genres": "Action Sci-Fi", "director": "Wachowski",
     "overview": "A hacker discovers reality is a simulation and joins a rebellion against machines."},
    {"title": "Inception", "genres": "Action Sci-Fi Thriller", "director": "Nolan",
     "overview": "A thief who steals secrets through dream-sharing technology is given a chance to erase his past."},
    {"title": "Interstellar", "genres": "Adventure Drama Sci-Fi", "director": "Nolan",
     "overview": "A team of explorers travel through a wormhole in space to ensure humanity's survival."},
    {"title": "The Dark Knight", "genres": "Action Crime Drama", "director": "Nolan",
     "overview": "Batman faces the Joker, a criminal mastermind who plunges Gotham into anarchy."},
    {"title": "John Wick", "genres": "Action Crime Thriller", "director": "Stahelski",
     "overview": "An ex-hitman comes out of retirement to track down the gangsters that took everything from him."},
    {"title": "The Avengers", "genres": "Action Adventure Sci-Fi", "director": "Whedon",
     "overview": "Earth's mightiest heroes assemble to stop an alien invasion led by a powerful villain."},
    {"title": "Titanic", "genres": "Drama Romance", "director": "Cameron",
     "overview": "A young couple from different social classes fall in love aboard a doomed ocean liner."},
    {"title": "The Notebook", "genres": "Drama Romance", "director": "Cassavetes",
     "overview": "A poor young man and a rich young woman fall passionately in love despite their families."},
    {"title": "Avatar", "genres": "Action Adventure Sci-Fi", "director": "Cameron",
     "overview": "A paraplegic marine on an alien planet becomes torn between orders and protecting the world."},
    {"title": "Gladiator", "genres": "Action Adventure Drama", "director": "Scott",
     "overview": "A betrayed Roman general fights his way back to seek revenge against a corrupt emperor."},
    {"title": "The Shawshank Redemption", "genres": "Drama", "director": "Darabont",
     "overview": "Two imprisoned men bond over years, finding solace and eventual redemption through decency."},
    {"title": "Se7en", "genres": "Crime Drama Thriller", "director": "Fincher",
     "overview": "Two detectives hunt a serial killer who uses the seven deadly sins as his motives."},
]


@st.cache_data
def load_similarity_matrix():
    df = pd.DataFrame(MOVIES)
    df["content"] = df.apply(
        lambda row: " ".join([
            row["overview"],
            (row["genres"] + " ") * 3,
            (row["director"] + " ") * 2,
        ]),
        axis=1,
    )
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(df["content"])
    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return df, similarity_matrix


def recommend(df, similarity_matrix, title, top_n=5):
    title_to_index = pd.Series(df.index, index=df["title"])
    idx = title_to_index[title]
    scores = list(enumerate(similarity_matrix[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    scores = [s for s in scores if s[0] != idx][:top_n]
    result = df.iloc[[i for i, _ in scores]][["title", "genres", "director"]].copy()
    result["similarity"] = [round(score, 3) for _, score in scores]
    return result.reset_index(drop=True)


st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="centered")
st.title("🎬 Movie Recommender")
st.write("Pick a movie you like and get similar recommendations based on genre, plot, and director.")

df, similarity_matrix = load_similarity_matrix()

selected_movie = st.selectbox("Choose a movie", df["title"].tolist())
top_n = st.slider("Number of recommendations", min_value=1, max_value=8, value=4)

if st.button("Recommend"):
    results = recommend(df, similarity_matrix, selected_movie, top_n)
    st.subheader(f"Because you liked '{selected_movie}':")
    for _, row in results.iterrows():
        st.markdown(f"**{row['title']}**  \n{row['genres']} · Directed by {row['director']}  \nSimilarity: {row['similarity']}")
        st.divider()
