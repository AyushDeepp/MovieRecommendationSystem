
import streamlit as st
import pandas as pd
import numpy as np

from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

movies=pd.read_csv('dataset/movies.csv')
ratings=pd.read_csv('dataset/ratings.csv')

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

def recommendation(movie_name,no_of_movies):
    movie_list = movies[movies['title'].str.contains(movie_name)]
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
            recommended_movies.append({'Title': movies.iloc[idx]['title'].values[0], 'Distance': val[1]})
        df = pd.DataFrame(recommended_movies, index=range(1, no_of_movies+1))
        return df
    else:
        return "Movie not found..."


st.title("Movie Recommendation System")

#selected_movie=st.selectbox("Select a movie", movie_titles_list)
selected_movie=st.text_input("Enter a Movie Name: ")
no_of_movies=st.text_input("No of Movies to Recommend: ")
if(st.button("Recommend: ")):
  st.text("Recommended Movies are: ")
  
  recommended_movies=recommendation(selected_movie,int(no_of_movies))
  st.table(recommended_movies)