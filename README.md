# Search-Movies
Search Engine for Movies 
## Problem Statement
Many of us face problem in remembering the names of the movies, but we always remember the basic plot and cast.

### Solution?
This search engine helps you find the name of the movie and also recommends movies that have similar plot and genre by using concepts of **Information Retrival**.

## Functionalities
 - Reads the .csv file of all movies with plots.
 - Removes all the characters that are not letters or numbers, converts the sentence to lowercase and removes the stopwords occuring in the document.
 - Removes all the documents that doesnot have plots and applies the necessary function to the documents that have plots.
 - Tokenizes all the words in a document's plot and stores them in a list, then it stores all the lists of individual plots into a corpus.
 - Creates a document vector by taking the mean of word vectors of the words in the document.
 - Asks for the query from the user.
 - Performs preprocessing(above mentioned operations) to the query.
 - Displays top ten movie titles that matched with the query given and displays it's corresponding document.

## Tech Stack
Technologies to be used.
 - NLTK(Natural Langauge Tool Kit Library)
 - Gensim Library(word2vec model)
 - Numpy and Pandas libraries
 - Regex Library 


## Team Members
 - [Nikhil Munigela](2017AAPS0418H)
 - [Jathin Badam](2017A3PS0495H)
 - [Venkata Sai Karthik Jagini](2017AAPS0371H)
 - [Nikhil Kandukuri](2017A3PS0497H)
