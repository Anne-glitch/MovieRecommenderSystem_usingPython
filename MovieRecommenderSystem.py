#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np 


# In[2]:


#tmdb csv files of movies and credits
df1 = pd.read_csv('tmdb_5000_movies.csv')
df2 = pd.read_csv('tmdb_5000_credits.csv')


# In[3]:


#merging the two files on the column labelled-"title"
df2 = df2.merge(df1, on = 'title')


# In[4]:


df2.head()


# In[ ]:


#Demographic Filtering


# In[5]:


#Calculating the average amount of votes received by the movies
C= df2['vote_average'].mean()
C


# In[6]:


#DEMOGRAPHIC FILTERING
#mean rating for all the movies is approx 6 on a scale of 10
#90th percentile as our cutoff.
m= df2['vote_count'].quantile(0.9)
m


# In[7]:


#filter out the movies that qualify for the chart
q_movies = df2.copy().loc[df2['vote_count'] >= m]
q_movies.shape


# In[8]:


#a function, weighted_rating() that defines a new feature score
def weighted_rating(x, m=m, C=C):
    v = x['vote_count']
    R = x['vote_average']
    # Calculation based on the IMDB formula
    return (v/(v+m) * R) + (m/(m+v) * C)


# In[9]:


# Define a new feature 'score' and calculate its value with `weighted_rating()`
q_movies['score'] = q_movies.apply(weighted_rating, axis=1)


# In[10]:


#sort the DataFrame based on the score feature and output the title, vote count, vote average and weighted rating or 
#score of the top 10 movies
q_movies = q_movies.sort_values('score', ascending=False)

#Print the top 10 movies
q_movies[['title', 'vote_count', 'vote_average', 'score']].head(10)


# In[11]:


#finding movies that are very popular by simply sorting the dataset by the popularity column.
pop= df2.sort_values('popularity', ascending=False)
import matplotlib.pyplot as plt
plt.figure(figsize=(12,4))

plt.barh(pop['title'].head(6),pop['popularity'].head(6), align='center',
        color='orchid')
plt.gca().invert_yaxis()
plt.xlabel("Popularity")
plt.title("Popular Movies")


# In[ ]:


#Content Based Filtering


# In[12]:


#Plot description based Recommender
df2['overview'].head(5)


# In[13]:


#Import TfIdfVectorizer from scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer

#Compute Term Frequency-Inverse Document Frequency (TF-IDF) vectors for each overview.
#Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
tfidf = TfidfVectorizer(stop_words='english')

#Replace NaN with an empty string
df2['overview'] = df2['overview'].fillna('')

#Construct the required TF-IDF matrix by fitting and transforming the data
tfidf_matrix = tfidf.fit_transform(df2['overview'])

#Output the shape of tfidf_matrix
tfidf_matrix.shape


# In[ ]:


#As we have used the TF-IDF vectorizer, calculating the dot product will directly give us the cosine similarity score. Therefore, we will use sklearn's linear_kernel() instead of cosine_similarities() since it is quicker.


# In[14]:


# Import linear_kernel
from sklearn.metrics.pairwise import linear_kernel

# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)


# In[15]:


#define a function that takes in a movie title as an input and outputs a list of the 10 most similar movies.
#need a mechanism to identify the index of a movie in our metadata DataFrame, given its title


# In[16]:


#Construct a reverse map of indices and movie titles
indices = pd.Series(df2.index, index=df2['title']).drop_duplicates()


# In[17]:


# Function that takes in movie title as input and outputs most similar movies
def get_recommendations(title, cosine_sim=cosine_sim):
    # Get the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwise similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return df2['title'].iloc[movie_indices]


# In[18]:


get_recommendations('The Dark Knight Rises')


# In[19]:


get_recommendations("Spectre")


# In[20]:


get_recommendations("The Avengers")


# In[21]:


get_recommendations("The Lord of the Rings: The Return of the King")


# In[ ]:


#Credits, Genres and Keywords Based Recommender


# In[22]:


# Parse the stringified features into their corresponding python objects
from ast import literal_eval

features = ['cast', 'crew', 'keywords', 'genres']
for feature in features:
    df2[feature] = df2[feature].apply(literal_eval)


# In[23]:


# Get the director's name from the crew feature. If director is not listed, return NaN
def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan


# In[24]:


# Returns the list top 3 elements or entire list; whichever is more.
def get_list(x):
    if isinstance(x, list):
        names = [i['name'] for i in x]
        #Check if more than 3 elements exist. If yes, return only first three. If no, return entire list.
        if len(names) > 3:
            names = names[:3]
        return names

    #Return empty list in case of missing/malformed data
    return []


# In[25]:


# Define new director, cast, genres and keywords features that are in a suitable form.
df2['director'] = df2['crew'].apply(get_director)

features = ['cast', 'keywords', 'genres']
for feature in features:
    df2[feature] = df2[feature].apply(get_list)


# In[26]:


# Print the new features of the first 3 films
df2[['title', 'cast', 'director', 'keywords', 'genres']].head(3)


# In[27]:


# Function to convert all strings to lower case and strip names of spaces
def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        #Check if director exists. If not, return empty string
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''


# In[28]:


# Apply clean_data function to features.
features = ['cast', 'keywords', 'director', 'genres']

for feature in features:
    df2[feature] = df2[feature].apply(clean_data)


# In[29]:


# here's the "metadata soup", which is a string that contains all the metadata that we want to feed to our vectorizer (namely actors, director and keywords).
def create_soup(x):
   return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])
df2['soup'] = df2.apply(create_soup, axis=1)


# In[30]:


# Import CountVectorizer and create the count matrix
from sklearn.feature_extraction.text import CountVectorizer

count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(df2['soup'])


# In[31]:


# Compute the Cosine Similarity matrix based on the count_matrix
from sklearn.metrics.pairwise import cosine_similarity

cosine_sim2 = cosine_similarity(count_matrix, count_matrix)


# In[32]:


# Reset index of our main DataFrame and construct reverse mapping as before
df2 = df2.reset_index()
indices = pd.Series(df2.index, index=df2['title'])


# In[33]:


get_recommendations('The Dark Knight Rises', cosine_sim2)


# In[34]:


get_recommendations('The Godfather', cosine_sim2)


# In[35]:


get_recommendations('Spectre', cosine_sim2)


# In[36]:


get_recommendations('Harry Potter and the Half-Blood Prince', cosine_sim2)


# In[37]:


get_recommendations('The Da Vinci Code', cosine_sim2)


# In[38]:


get_recommendations("The Avengers", cosine_sim2)


# In[ ]:


#Collaborative Filtering


# In[39]:


from surprise import Reader, Dataset, SVD
reader = Reader()
ratings = pd.read_csv('ratings_small.csv')
#this dataset movies are rated on a scale of 5 unlike the earlier one.
ratings.head()


# In[40]:


from surprise.model_selection import cross_validate

# Load the dataset (download it if needed)
data = Dataset.load_builtin('ml-100k')

# Use the famous SVD algorithm
algo = SVD()

# Run 5-fold cross-validation and then print results
cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)


# In[41]:


data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
svd = SVD()
cross_validate(svd, data, measures=['RMSE', 'MAE'], cv=10, verbose=True)


# In[42]:


trainset = data.build_full_trainset()
svd.fit(trainset)


# In[43]:


ratings[ratings['userId'] == 1]


# In[44]:


svd.predict(1, 302, 3)


# In[ ]:


#For movie with ID 302, we get an estimated prediction of 2.618. One startling feature of this recommender system is that it doesn't care what the movie is (or what it contains). It works purely on the basis of an assigned movie ID and tries to predict ratings based on how the other users have predicted the movie.

