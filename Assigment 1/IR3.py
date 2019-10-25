#Importing necessary modules
import os
import nltk
from nltk.corpus import stopwords
from gensim.models import word2vec
import re
import numpy as np
import pandas as pd

def clean_sentence(val):
	"This function Removes all the characters that are not letters or numbers , converts the sentence to lowercase and removes the stopwords occuring in the document"		
	regex = re.compile('([^\s\w]|_)+')		#Removes chars that are not letters or numbers
	sentence = regex.sub('', val).lower()	#Converts the sentence to lowercase
	sentence = sentence.split(" ")
		
	for word in list(sentence):
		if word in eng_stopwords :
			sentence.remove(word)			#Removes stop words that occur in the sentence  
	
	sentence = " ".join(sentence)
	return sentence

def clean_dataframe(data):
	"This function Removes all the documents that doesnot have plots and applies the clean_sentence function to the documents that have plots"	
	data = data.dropna(how='any')						#Drops the NaNs occuring in the dataframe 
	data['Plot'] = data['Plot'].apply(clean_sentence)	#Applies the clean sentence function
	
	return data

def build_corpus(data):
	"This function Tokenizes all the words in a document's plot and stores them in a list, then it stores all the lists of individual plots into a corpus"
	corpus = []
		
	for sentence in data['Plot'].iteritems():
		word_list = sentence[1].split(" ")		#Tokenizes words of each plot and appends them 
		corpus.append(word_list)				#to a list which is further appended to corpus
			
	return corpus

def documentvec(word2vec_model,plotwords):
	"This function Creates a document vector by taking the mean of word vectors of the words in the document"
	k=[]
	for i in range(len(plotwords)): 
		if plotwords[i] in word2vec_model.wv.vocab:		#model.wv.vocab gives the entire word vocabulary 
			k.append(word2vec_model[plotwords[i]])		#of the generated model upon the given dataset
	return np.mean(k,axis=0)


	
#Loading the CSV file containing the movies and plots dataset
data_movies = pd.read_csv(r'/home/jathin/Desktop/IR_Project/wiki_movie_plots_deduped.csv')	 

nltk.download('stopwords')
eng_stopwords = stopwords.words('english')		#Loading the stopwords in english language for further preprocesing

cleaned_data = clean_dataframe(data_movies)		#Applying clean_dataframe function to the loaded CSV file

corpus = build_corpus(cleaned_data)				#Applying build_corpus function to the cleaned dataframe


#Generating the word2vec model out of the inputted corpus
model = word2vec.Word2Vec(corpus, size=100, window=20, min_count=10, workers=4)	

docvecs=[]
for i in range(len(corpus)): 
		docvecs.append(documentvec(model,corpus[i]))	#Generating a document vector for each of the list item 
														#in corpus, which itself is a list of tokens corresponding to a plot 

			
query=input('Enter any key plot-points in the movie:')		#Taking query from the user
mod_query=clean_sentence(query)			#Processing the user's query just like the dataset


query_wordlist=mod_query.split(' ')


p=documentvec(model,query_wordlist)		#Creating document vector for user's query

	
from scipy.spatial import distance
result=[]
for i in range(len(docvecs)):
	result.append(1-distance.cosine(p,docvecs[i]))		#Calculating cosine distance of the query's doc vector 
														#with the list of all the plot doc vectors in docvecs list

final_results = np.array(result)				#Forming an array of the computed cosine distances

top_results = (-final_results).argsort()[:10]	#Sorting and filtering out top 10 results of the cosine distances and storing in idx
	

print('The movie you are looking for might be one of these:\n')
#Returning the corresponding title and plot of the movies sorted according to their cosine distances from the initial dataset
print(cleaned_data.iloc[top_results,[1,7]])