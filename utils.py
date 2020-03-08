from scipy import spatial, sparse
from scipy.stats import chi2
from num2words import num2words
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from bs4 import BeautifulSoup as bs

import copy
import nltk
import scipy
import multiprocessing

import numpy as np
import pandas as pd

stop_words = stopwords.words('english')
lemmatizer = nltk.stem.WordNetLemmatizer()
stemmer = nltk.stem.PorterStemmer()
w_tokenizer = nltk.tokenize.WhitespaceTokenizer()

def get_pos_tags(text):
    pos = nltk.pos_tag(text)

    m = {i:0 for i in [j[1] for j in pos]}
    m.keys
    for _, j in pos:
        m[j] += 1
    return m