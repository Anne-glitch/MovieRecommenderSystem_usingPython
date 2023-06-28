# MovieRecommenderSystem_usingPython
Here's a Movie Recommendation System using three different filtering techniques. The datasets  are from Kaggle.
## Introduction

Movie recommendation systems are a great way to help users discover new movies that they love. These systems are used in a lot of different applications, from streaming services to e-commerce websites. In this project, there are three types of movie recommendation systems: Demographic Filtering, Content-Based Filtering, and Collaborative Filtering.

## Demographic Filtering

Demographic filtering is a technique that uses user demographic data to recommend movies. This approach assumes that users with similar demographics will like similar movies. For example, a 25-year-old male who likes action movies may be recommended an action movie that was liked by other 25-year-old men who also enjoy action movies. Demographic filtering is easy to implement but has limited accuracy since it does not take into account the personal tastes of individual users. The general idea of this system is that the movies that are more popular and critically acclaimed will have a higher probability of being liked by the average audience

Using IMDB's weighted rating (wr) which is given as:-

(WR) = (v ÷ (v+m)) × R + (m ÷ (v+m)) × C

where,

v is the number of votes for the movie;
m is the minimum votes required to be listed in the chart;
R is the average rating of the movie; And
C is the mean vote across the whole report

## Content-Based Filtering

Content-based filtering is a technique that recommends movies based on the similarity of their content. This approach uses features of the movie such as genre, actors, director, and plot summary to recommend movies that are similar to movies the user has already liked. For example, if a user has watched and enjoyed the movie "Spectre" the system may recommend similar action movies with Daniel Craig as the lead actor and Sam Mendes as the director. Content-based filtering is more accurate than demographic filtering since it takes into account user preferences, but it can be limited by the quality of the movie features used.

Term Frequency-Inverse Document Frequency (TF-IDF) vectors
Term frequency is the relative frequency of a word in a document and is given as (term instances/total instances). Inverse Document Frequency is the relative count of documents containing the term given as log(number of documents/documents with the term) The overall importance of each word to the documents in which they appear is equal to TF * IDF

similarity score- the Euclidean, the Pearson, and the cosine similarity scores.
cosine similarity to calculate a numeric quantity that denotes the similarity between two movies. It is independent of magnitude and is relatively easy and fast to calculate.

$Cos(x, y) = x . y / ||x|| * ||y||$

### Credits, Genres, and Keywords-Based Recommender

We are going to build a recommender based on the following metadata: the top 3 actors, director, related genres, and the movie plot keywords that will help improve the quality of the recommender with the usage of better metadata."Metadata soup", is a string that contains all the metadata that we want to feed to our vectorizer (namely actors, director, and keywords). One important difference is that we use the CountVectorizer() instead of TF-IDF. This is because we do not want to weigh it down with the factor of whether the actor/director has acted or directed in other movies.

## Collaborative Filtering

Collaborative filtering is a technique that recommends movies based on the similarity of user preferences. This approach uses data from multiple users to find patterns in movie preferences and make recommendations. Collaborative filtering can be divided into two types: user-based and item-based. User-based collaborative filtering recommends movies based on the preferences of similar users, while item-based collaborative filtering recommends movies that are similar to those the user has already liked. Collaborative filtering is more accurate than demographic and content-based filtering since it takes into account the preferences of individual users, but it can be limited by the availability of user data.

It is again of two types:-

1. User-based filtering
2. Item-based filtering

## Datasets Used In This Project

****TMDB 5000 Movie Dataset-**** tmdb_5000_movies.csv and  • tmdb_5000_credits.csv

**The Movies Dataset- ratings_small.csv:** The subset of 100,000 ratings from 700 users on 9,000 movies.

## Advantages and Disadvantages of each approach

### Demographic Filtering

Demographic filtering is the simplest approach to movie recommendation systems. It is easy to implement and does not require any data about individual user preferences. However, it has limited accuracy since it assumes that users with similar demographics will have similar movie preferences. This may not always be true, as individual tastes can vary widely regardless of demographics.

### Content-Based Filtering

Content-based filtering is more accurate than demographic filtering since it takes into account user preferences. It uses movie features such as genre, actors, director, and plot summary to recommend movies that are similar to movies the user has already liked. However, it can be limited by the quality of the movie features used. If the features are not well-defined or are too general, the recommendations may not be very accurate.

### Collaborative Filtering

Collaborative filtering is the most accurate approach to movie recommendation systems since it takes into account the preferences of individual users. It uses data from multiple users to find patterns in movie preferences and make recommendations. However, it can be limited by the availability of user data. If there is not enough data about user preferences, the recommendations may not be very accurate.

## Applications of Movie Recommender Systems

Movie recommendation systems are used in a wide variety of applications. Streaming services such as Netflix, Amazon Prime Video, and Hulu all use recommendation systems to suggest new movies to their subscribers. E-commerce websites such as Amazon and Best Buy also use recommendation systems to suggest movies to their customers. These systems can help increase customer engagement and satisfaction by offering personalized recommendations.

## Conclusion

Movie recommendation systems are a valuable tool for helping users discover new movies they love. Demographic filtering, content-based filtering, and collaborative filtering are three different approaches to movie recommendation systems, each with its own advantages and disadvantages. Demographic filtering is easy to implement but has limited accuracy. Content-based filtering is more accurate than demographic filtering but can be limited by the quality of movie features used. Collaborative filtering is the most accurate approach, but it can be limited by the availability of user data. Ultimately, the best approach to movie recommendation systems depends on the specific application and the available data. The best approach to movie recommendation systems depends on the specific application and the available data. Regardless of the approach used, movie recommendation systems can be a powerful tool for increasing customer engagement and satisfaction.
