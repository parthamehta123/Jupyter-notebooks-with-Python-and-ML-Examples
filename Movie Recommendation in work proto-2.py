# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 22:28:13 2018

@author: Admin
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.sparse as sparse
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


ratings = pd.read_csv("ml-latest-small/ml-latest-small/ratings.csv")
ratings.head()


# In[4]:


ratings.userId.nunique(), ratings.movieId.nunique()


# In[5]:


ratings = ratings[["userId", "movieId", "rating"]]

data = ratings.groupby("userId", as_index=False).agg({"movieId": 'count'})
data.head()


# In[6]:


data.movieId.hist()


# In[7]:


data.movieId.describe()


# In[10]:


movies = pd.read_csv("ml-latest-small/ml-latest-small/movies.csv")
movies.head()


# In[11]:


users = list(np.sort(ratings.userId.unique())) # Get our unique customers
movies = list(ratings.movieId.unique()) # Get our unique products that were purchased
rating = list(ratings.rating) # All of our purchases


# In[12]:


rows = ratings.userId.astype("category", categories = users).cat.codes
# Get the associated row indices
cols = ratings.movieId.astype("category", categories = movies).cat.codes
# Get the associated column indices
user_item = sparse.csr_matrix((rating, (rows, cols)), shape=(len(users), len(movies)))


# In[13]:


matrix_size = user_item.shape[0]*user_item.shape[1] # Number of possible interactions in the matrix
num_purchases = len(user_item.nonzero()[0]) # Number of items interacted with
sparsity = 100*(1-(1.0*num_purchases/matrix_size))
sparsity


# In[14]:


user_item


# In[15]:


import implicit


# In[16]:


model = implicit.als.AlternatingLeastSquares(factors=10, 
                                             iterations=20, 
                                             regularization=0.1, 
                                             num_threads=4)
model.fit(user_item.T)


# In[17]:


movies_table = pd.read_csv("ml-latest-small/ml-latest-small/movies.csv")
movies_table.head()


# In[18]:


def similar_items(item_id, movies_table, movies, N=5):
    """
    Input
    -----
    
    item_id: int
        MovieID in the movies table
    
    movies_table: DataFrame
        DataFrame with movie ids, movie title and genre
        
    movies: np.array
        Mapping between movieID in the movies_table and id in the item user matrix
        
    N: int
        Number of similar movies to return
        
    Output
    -----
    df: DataFrame
        DataFrame with selected movie in first row and similar movies for N next rows
    """
    # Get movie user index from the mapping array
    user_item_id = movies.index(item_id)
    # Get similar movies from the ALS model
    similars = model.similar_items(user_item_id, N=N+1)    
    # ALS similar_items provides (id, score), we extract a list of ids
    l = [item[0] for item in similars]
    # Convert those ids to movieID from the mapping array
    ids = [movies[ids] for ids in l]
    # Make a dataFrame of the movieIds
    ids = pd.DataFrame(ids, columns=['movieId'])
    # Add movie title and genres by joining with the movies table
    recommendation = pd.merge(ids, movies_table, on='movieId', how='left')
    
    return recommendation


# In[19]:


df = similar_items(10, movies_table, movies, 5)
df


# In[23]:


df = similar_items(502, movies_table, movies, 5)
df


# In[22]:


df = similar_items(1, movies_table, movies, 5)
df


# In[23]:


metadata = pd.read_csv('m1-latest-small/m1-latest-small/movies_metadata.csv')
metadata.head(2)


# In[24]:


image_data = metadata[['imdb_id', 'poster_path']]
image_data.head()


# In[25]:


links = pd.read_csv("m1-latest-small/m1-latest-small/links.csv")
links.head()


# In[26]:


links = links[['movieId', 'imdbId']]


# In[27]:


image_data = image_data[~ image_data.imdb_id.isnull()]


# In[28]:


def app(x):
    try:
        return int(x[2:])
    except ValueError:
        print(x)


# In[29]:


image_data['imdbId'] = image_data.imdb_id.apply(app)
image_data = image_data[~ image_data.imdbId.isnull()]
image_data.imdbId = image_data.imdbId.astype(int)
image_data = image_data[['imdbId', 'poster_path']]
image_data.head()


# In[30]:


posters = pd.merge(image_data, links, on='imdbId', how='left')

posters = posters[['movieId', 'poster_path']]

posters = posters[~ posters.movieId.isnull()]

posters.movieId = posters.movieId.astype(int)

posters.head()


# In[31]:


movies_table = pd.merge(movies_table, posters, on='movieId', how='left')
movies_table.head()


# In[32]:


from  IPython.display  import HTML
from IPython.display import display

def display_recommendations(df):
    
    images = ''
    for ref in df.poster_path:
            if ref != '':
                link = 'http://image.tmdb.org/t/p/w185/' + ref
                images += "<img style='width: 120px; margin: 0px;                   float: left; border: 1px solid black;' src='%s' />"               % link
    display(HTML(images))


# In[33]:


df = similar_items(500, movies_table, movies, 5)
display_recommendations(df)


# In[34]:


def similar_and_display(item_id, movies_table, movies, N=5):
    
    df = similar_items(item_id, movies_table, movies, N=N)
    
    display_recommendations(df)


# In[35]:


similar_and_display(1, movies_table, movies, 5)