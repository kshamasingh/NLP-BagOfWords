#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 23:35:52 2021

@author: kshama.singh
"""

import nltk
# Cleaning the texts
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer


paragraph = open("../BagOfWords/BagOfWords.txt", 'r').read()
#print(paragraph)            
        

# Preparing the sentences
def processData_SentencesTokenize(paragraph):
    sentences = nltk.sent_tokenize(paragraph)
    return sentences    


# Preparing the dataset
def processData_dataset(sentences, porterStemmer):
    corpus = []
    for i in range(len(sentences)):
        review = re.sub('[^a-zA-Z]', ' ', sentences[i])
        review = review.lower()
        review = review.split()
        review = [porterStemmer.stem(word) for word in review if not word in set(stopwords.words('english'))]
        review = ' '.join(review)
        corpus.append(review)
    return corpus


# Creating the Bag of Words model
def createBagOfWords(corpus):
    cv = CountVectorizer(max_features = 1500)
    return cv.fit_transform(corpus).toarray()
    


porterStemmer = PorterStemmer()
wordnet = WordNetLemmatizer()
sentences = processData_SentencesTokenize(paragraph)
corpus = processData_dataset(sentences, porterStemmer);
BagOfWords = createBagOfWords(corpus);

