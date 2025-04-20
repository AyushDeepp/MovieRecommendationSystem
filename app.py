import streamlit as st
import pandas as pd
import numpy as np
import requests

from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

movies=pd.read_csv('dataset/movies.csv')
ratings=pd.read_csv('dataset/ratings.csv')

# Add after loading movies dataset
all_genres = set()
for genres in movies['genres'].str.split('|'):
    all_genres.update(genres)
all_genres = sorted(list(all_genres))

a = movies['title']
movie_titles_list = a.to_list()

dataset = ratings.pivot(index="movieId",columns="userId",values="rating")
dataset.fillna(0,inplace=True)

num_user_voted = ratings.groupby('movieId')['rating'].agg('count')
num_movies_voted = ratings.groupby('userId')['rating'].agg('count')

final_dataset = dataset.loc[num_user_voted[num_user_voted > 10].index, :]
final_dataset = final_dataset.loc[: , num_movies_voted[num_movies_voted>50].index]

csr_data = csr_matrix(final_dataset.values)
final_dataset.reset_index(inplace=True)

knn = NearestNeighbors(metric='cosine', algorithm = 'brute', n_neighbors = 30, n_jobs = -1)
knn.fit(csr_data)

def get_movie_info(title):
    """Fetch basic movie info from OMDB API"""
    # You'll need to sign up for a free API key at http://www.omdbapi.com/
    API_KEY = "448e92b8"
    url = f"http://www.omdbapi.com/?t={title}&apikey={API_KEY}"
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if data.get('Response') == 'True':
                return {
                    'Year': data.get('Year', 'N/A'),
                    'Rating': data.get('imdbRating', 'N/A'),
                    'Plot': data.get('Plot', 'N/A'),
                    'Poster': data.get('Poster', 'N/A')
                }
    except:
        pass
    return None

# Modify the recommendation function
def recommendation(movie_name, no_of_movies, genres=None):
    movie_list = movies[movies['title'].str.contains(movie_name)]
    if not len(movie_list):
        return "Movie not found..."
        
    print(type(movie_list))
    if len(movie_list):
        movie_index = movie_list.iloc[0]['movieId']
        movie_index = final_dataset[final_dataset['movieId'] == movie_index].index[0]

        distances, indices = knn.kneighbors(csr_data[movie_index], n_neighbors=no_of_movies+1)

        indices_list = indices.squeeze().tolist()
        distances_list = distances.squeeze().tolist()
        index_distance_pairs = list(zip(indices_list, distances_list))
        rec_movies_indices = sorted(index_distance_pairs[1:], key=lambda x: x[1], reverse=True)

        recommended_movies = []
        for val in rec_movies_indices:
            movie_index = final_dataset.iloc[val[0]]['movieId']
            idx = movies[movies['movieId'] == movie_index].index
            title = movies.iloc[idx]['title'].values[0]
            info = get_movie_info(title)
            
            movie_data = {
                'Title': title,
                'Similarity Score': f"{(1 - val[1]) * 100:.1f}%"
            }
            if info:
                movie_data.update({
                    'Year': info['Year'],
                    'Rating': info['Rating'],
                    'Plot': info['Plot'],
                    'Poster': info['Poster']
                })
            recommended_movies.append(movie_data)
        
        # Filter by selected genres
        if genres:
            filtered_movies = []
            for movie in recommended_movies:
                movie_genres = movies[movies['title'] == movie['Title']]['genres'].iloc[0].split('|')
                if any(genre in movie_genres for genre in genres):
                    filtered_movies.append(movie)
            recommended_movies = filtered_movies

        # Ensure we don't try to create more indices than we have movies
        actual_movies = min(len(recommended_movies), no_of_movies)
        if actual_movies == 0:
            return "No movies found matching selected genres..."
        
        # Create DataFrame with correct number of indices
        df = pd.DataFrame(recommended_movies[:actual_movies], 
                         index=range(1, actual_movies + 1))
        return df
    else:
        return "Movie not found..."

def validate_inputs(movie_name, num_movies):
    if not movie_name:
        return False, "Please enter a movie name"
    if not num_movies:
        return False, "Please enter number of movies"
    try:
        num = int(num_movies)
        if num <= 0 or num > 30:
            return False, "Please enter a number between 1 and 30"
    except ValueError:
        return False, "Please enter a valid number"
    return True, ""

st.title("Movie Recommendation System")
st.write("Find movies similar to your favorites!")

st.sidebar.header("Filters")
selected_genres = st.sidebar.multiselect(
    "Filter by Genres:",
    options=all_genres
)

st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stButton button {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
    }
    .movie-card {
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .similarity-score {
        color: #ff4b4b;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    selected_movie = st.text_input("Enter a Movie Name:", 
                                  help="Type part of the movie name")
with col2:
    no_of_movies = st.number_input("Number of Recommendations:", 
                                  min_value=1, max_value=30, value=5)

if st.button("Get Recommendations"):
    is_valid, error_message = validate_inputs(selected_movie, no_of_movies)
    if not is_valid:
        st.error(error_message)
    else:
        with st.spinner('Finding recommendations...'):
            recommended_movies = recommendation(selected_movie, int(no_of_movies), selected_genres)
            if isinstance(recommended_movies, str):
                st.error(recommended_movies)
            else:
                st.success("Here are your recommendations!")
                st.dataframe(recommended_movies, use_container_width=True)