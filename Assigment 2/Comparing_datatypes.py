#!/usr/bin/env python
# coding: utf-8

# In[1]:




#Importing necessary modules
import os
import nltk
from nltk.corpus import stopwords
#from gensim.models import word2vec
import re
import numpy as np
import pandas as pd 
from numpy import linalg as LA
#from future import division
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

num_hashes = 40; #no of hash functions
num_docs = 1000  #no of documents
currentshingleID = 0
docs_Shingle_Sets = {};
doc_names = []
totalshingle_count= 0
for i in range(0, num_docs):
  words = final_data.iloc[i,8].split(" ") 
  doc_names.append(final_data.iloc[i,0])
  shinglesin_doc = set() #We use set function to avoid repetations
  for index in range(0, len(words) - 2):
    shingle = words[index] + " " + words[index + 1] + " " + words[index + 2] #shingles of 3 consecutive words 
    b_shingle = bytes(shingle, encoding="utf-8")   #converting to bytes
    crc = binascii.crc32(b_shingle) & 0xffffffff   #converting to 32bit ascii value
    shinglesin_doc.add(crc)  #Converted Hash gets added if not present
  docs_Shingle_Sets[final_data.iloc[i,0]] = shinglesin_doc   #Adds the shingles to the correspondent document dictionary
  # Count the number of shingles across all documents.
  totalshingle_count= totalshingle_count+ (len(words) - 2)   #Total no. of shingles.
print (totalshingle_count/ num_docs)
max_shingleID = 2**32-1
nextPrime = 4294967311        #largest prime for 32 bit number


# In[2]:


max_shingleID = 2**32-1
nextPrime = 4294967311        #largest prime for 32 bit number
def random_coefficients(k):   #function to generate random values
  rand_list = []
  while k > 0:
    randIndex = random.randint(0, max_shingleID) 
    while randIndex in rand_list:
      randIndex = random.randint(0, max_shingleID) 
    rand_list.append(randIndex)
    k = k - 1
  return rand_list
coeffA = random_coefficients(num_hashes)
coeffB = random_coefficients(num_hashes)


# In[3]:


print ('\nGenerating MinHash signature_lists_list for all documents...')
signature_lists_list = []    #signature matrix 
for docID in doc_names:
  shingleIDSet = docs_Shingle_Sets[docID]
  signature_list = []        #list for each hash function
  # For each of the random hash functions...
  for i in range(0, num_hashes):
    minHashCode = nextPrime + 1
    for shingleID in shingleIDSet:
      hashCode = (coeffA[i] * shingleID + coeffB[i]) % nextPrime   #hash fucntions
      if hashCode < minHashCode:
        minHashCode = hashCode
    signature_list.append(minHashCode)
  signature_lists_list.append(signature_list)


# In[4]:


def dot(K, L):  #dot product function
   if len(K) != len(L):
      return 0

   return sum(i[0] * i[1] for i in zip(K, L))


# In[11]:


numElems = int(num_docs * (num_docs - 1) / 2)
JSim = [0 for x in range(numElems)]
estJSim = [0 for x in range(numElems)]
def getTriangleIndex(i, j):
  if i == j:
    sys.stderr.write("Can't access triangle matrix with i == j")
    sys.exit(1)
  if j < i:
    temp = i
    i = j
    j = temp
  k = int(i * (num_docs - (i + 1) / 2.0) + j - i) - 1
  return k

if num_docs <= 2500:
    print ("\nCalculating Jaccard Similarities...")
    for i in range(0, num_docs):
      if (i % 100) == 0:
        print ("  (" + str(i) + " / " + str(num_docs) + ")")
      s1 = docs_Shingle_Sets[doc_names[i]] 
      for j in range(i + 1, num_docs):
        s2 = docs_Shingle_Sets[doc_names[j]]
        JSim[getTriangleIndex(i, j)] = J = (len(s1.intersection(s2)) / len(s1.union(s2))) 
del JSim               

print ('\nComparing all signatures...')
for i in range(0, num_docs):
  signature1 = signature_lists_list[i]
  for j in range(i + 1, num_docs):
    signature2 = signature_lists_list[j]
    count  = 0 
    for k in range(0, num_hashes):
      count = count + (signature1[k] == signature2[k])  
    estJSim[getTriangleIndex(i, j)] = (count / num_hashes)


threshold = 0.5
print ("\nList of Document Pairs with J(d1,d2) more than", threshold)
print ("Values shown are the estimated Jaccard similarity and the actual")
print ("Jaccard similarity.\n")
print ("                   Est. J   Act. J")

for i in range(0, num_docs):  
  for j in range(i + 1, num_docs):
    estJ = estJSim[getTriangleIndex(i, j)]
    if estJ > threshold:
      s1 = docs_Shingle_Sets[doc_names[i]]
      s2 = docs_Shingle_Sets[doc_names[j]]
      J = (len(s1.intersection(s2)) / len(s1.union(s2))) #actualy jaccard index
      print ("  %5s --> %5s   %.2f     %.2f" % (doc_names[i], doc_names[j], estJ, J))


# In[14]:


numElems = int(num_docs * (num_docs - 1) / 2)
CSim = [0 for x in range(numElems)]
estCSim = [0 for x in range(numElems)]
def getTriangleIndex(i, j):
  if i == j:
    sys.stderr.write("Can't access triangle matrix with i == j")
    sys.exit(1)
  if j < i:
    temp = i
    i = j
    j = temp
  k = int(i * (num_docs - (i + 1) / 2.0) + j - i) - 1
  return k

if num_docs <= 2500:
    print ("\nCalculating cosine Similarities...")
    for i in range(0, num_docs):
      if (i % 100) == 0:
        print ("  (" + str(i) + " / " + str(num_docs) + ")")
      s1 = docs_Shingle_Sets[doc_names[i]] 
      for j in range(i + 1, num_docs):
        s2 = docs_Shingle_Sets[doc_names[j]]
        s1 = list(s1)
        s2 = list(s2)
        mag1 = LA.norm(s1)
        mag2 = LA.norm(s2)
        mag1 = int(float(mag1))
        mag2 = int(float(mag2))
        CSim[getTriangleIndex(i, j)] = dot(s1,s2)/ (mag1 * mag2)
del CSim

print ('\nComparing all signatures...')
for i in range(0, num_docs):
  signature1 = signature_lists_list[i]
  for j in range(i + 1, num_docs):
    signature2 = signature_lists_list[j]
    mags1 = LA.norm(signature1)  #calculation of magnitude
    mags2 = LA.norm(signature2)
    mags1 = int(float(mags1))
    mags2 = int(float(mags2))
    estCSim[getTriangleIndex(i,j)] = dot(signature1, signature2) / (mags1 * mags2)
threshold = 0.6
print ("\nList of Document Pairs with J(d1,d2) more than", threshold)
print ("Values shown are the estimated cosine similarity and the actual")
print ("Cosine similarity.\n")
print ("                   Est. C   Act. C")

for i in range(0, num_docs):  
  for j in range(i + 1, num_docs):
    estC = estCSim[getTriangleIndex(i, j)]
    if estC > threshold:
      s1 = docs_Shingle_Sets[doc_names[i]]
      s2 = docs_Shingle_Sets[doc_names[j]]
      s1 = list(s1)
      s2 = list(s2)  #converting to list from set
      mag1 = LA.norm(s1)
      mag2 = LA.norm(s2)
      mag1 = int(float(mag1))
      mag2 = int(float(mag2))  
      C  = dot(s1,s2) / (mag1 * mag2)  
      print ("  %5s --> %5s   %.2f     %.2f" % (doc_names[i],doc_names[j], estC, C))


# In[ ]:


numElems = int(num_docs * (num_docs - 1) / 2)
ESim = [0 for x in range(numElems)]
estESim = [0 for x in range(numElems)]
def getTriangleIndex(i, j):   #to get single value from i & j values
  if i == j:
    sys.stderr.write("Can't access triangle matrix with i == j")
    sys.exit(1)
  if j < i:
    temp = i
    i = j
    j = temp
  k = int(i * (num_docs - (i + 1) / 2.0) + j - i) - 1
  return k

if num_docs <= 2500:
    print ("\nCalculating euclidean Similarities...")
    for i in range(0, num_docs):
      if (i % 100) == 0:
        print ("  (" + str(i) + " / " + str(num_docs) + ")")
      s1 = docs_Shingle_Sets[doc_names[i]] 
      for j in range(i + 1, num_docs):
        s2 = docs_Shingle_Sets[doc_names[j]]
        s1 = list(s1)
        s2 = list(s2)
        ESim[getTriangleIndex(i, j)] = sum((p-q)*2 for p, q in zip(s1, s2)) * .5 
del ESim

print ('\nComparing all signature_lists_list...')
for i in range(0, num_docs):
  signature1 = signature_lists_list[i]
  for j in range(i + 1, num_docs):
    signature2 = signature_lists_list[j]
    estESim[getTriangleIndex(i,j)] = sum((p-q)*2 for p, q in zip(signature1, signature2)) * .5 
threshold = 0.001
print ("\nList of Document Pairs with J(d1,d2) more than", threshold)
print ("Values shown are the estimated euclidean similarity and the actual")
print ("euclidean similarity.\n")
print ("                   Est. E   Act. E")

for i in range(0, num_docs):  
    for j in range(i + 1, num_docs):
      estE = estESim[getTriangleIndex(i, j)]
#     if estE > threshold:
      s1 = docs_Shingle_Sets[doc_names[i]]
      s2 = docs_Shingle_Sets[doc_names[j]]
      s1 = list(s1)
      s2 = list(s2)
      E = sum((p-q)*2 for p, q in zip(s1, s2)) * .5 
      print ("  %5s --> %5s   %.2f     %.2f" % (doc_names[i], doc_names[j], estE, E)) 
      #estE is the estimated value
      #E is the actual value


# In[ ]:





