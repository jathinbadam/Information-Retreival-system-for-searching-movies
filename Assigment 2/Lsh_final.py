#Importing necessary modules
import os
import nltk
from nltk.corpus import stopwords
from gensim.models import word2vec
import re
import numpy as np
import pandas as pd
import re
import time
import binascii
from bisect import bisect_right
from heapq import heappop, heappush
import random
import sys

def clean_sentence(val):
    "This function Removes all the characters that are not letters or numbers , converts the sentence to lowercase and removes the stopwords occuring in the document"      
    regex = re.compile('([^\s\w]|_)+')      #Removes chars that are not letters or numbers
    sentence = regex.sub('', val).lower()   #Converts the sentence to lowercase
    sentence = sentence.split(" ")
        
    for word in list(sentence):
        if word in eng_stopwords :
            sentence.remove(word)           #Removes stop words that occur in the sentence  
    
    sentence = " ".join(sentence)
    return sentence

def clean_dataframe(data):
    "This function Removes all the documents that doesnot have plots and applies the clean_sentence function to the documents that have plots"  
    data = data.dropna(how='any')                       #Drops the NaNs occuring in the dataframe 
    data['Plot'] = data['Plot'].apply(clean_sentence)   #Applies the clean sentence function
    
    return data



data_movies = pd.read_csv(r'/home/jathin/Desktop/Tech Stuff/IR_Project/IR_final/wiki_movie_plots_deduped.csv')   
nltk.download('stopwords')
eng_stopwords = stopwords.words('english')      #Loading the stopwords in english language for further preprocesing

cleaned_data = clean_dataframe(data_movies)     #Applying clean_dataframe function to the loaded CSV file

final_data = cleaned_data[0:1000]
final_data.insert(0,'docID',range(0,1000))
numHashes = 1000;
numDocs = 1000
final_data

print ("Shingling articles...")

curShingleID = 0

docsAsShingleSets = {};

docNames = []

totalShingles = 0

for i in range(0, numDocs):
  
  words = final_data.iloc[i,8].split(" ") 
  
  docNames.append(final_data.iloc[i,0])

  shinglesInDoc = set()

  for index in range(0, len(words) - 2):
        
    shingle = words[index] + " " + words[index + 1] + " " + words[index + 2]
    
    b_shingle = bytes(shingle, encoding="utf-8")

    crc = binascii.crc32(b_shingle) & 0xffffffff
    
    shinglesInDoc.add(crc)

  docsAsShingleSets[final_data.iloc[i,0]] = shinglesInDoc
  
  # Count the number of shingles across all documents.
  totalShingles = totalShingles + (len(words) - 2)
 
print (totalShingles / numDocs)
docNames

maxShingleID = 2**32-1

nextPrime = 4294967311

def pickRandomCoeffs(k):
  randList = []
  
  while k > 0:
    randIndex = random.randint(0, maxShingleID) 

    while randIndex in randList:
      randIndex = random.randint(0, maxShingleID) 

    randList.append(randIndex)
    k = k - 1
    
  return randList
 
coeffA = pickRandomCoeffs(numHashes)
coeffB = pickRandomCoeffs(numHashes)


signatures = {}
for docID in docNames:
    shingleIDSet = docsAsShingleSets[docID]
    signature = []
  # For each of the random hash functions...
    for i in range(0, numHashes):
        minHashCode = nextPrime + 1
        for shingleID in shingleIDSet:
          hashCode = (coeffA[i] * shingleID + coeffB[i]) % nextPrime 
          if hashCode < minHashCode:
            minHashCode = hashCode
        signature.append(minHashCode)
    signatures[docID] = signature 

signatures


class LSH:
    def __init__(self, b, k):
        self.b = b
        self.k = k
        self.r = 2
        self.buckets = [{} for i in range(b)]

    # update
    def update(self, docid, sig):
        for i in range(self.b):
            hash_value = hash(tuple(sig[i*self.r:(i+1)*self.r]))
            if hash_value not in self.buckets[i]:
                self.buckets[i][hash_value] = set()
                self.buckets[i][hash_value].add(docid)
            else:
                self.buckets[i][hash_value].add(docid)
                              
    #query
    def query(self, sig, docid):
        candidates = set()
        for i in range(self.b):
            hash_value = hash(tuple(sig[i*self.r:(i+1)*self.r]))
            if hash_value in self.buckets[i]:
                candidates.update(self.buckets[i][hash_value]);
        if docid in candidates:
            candidates.remove(docid)
        return candidates

lsh = LSH(500, 1000)
for key, value in signatures.items():
    lsh.update(key,value)

key = 210
candidates = lsh.query(signatures[key], key)
candidates

len(buckets)

lsh.buckets

