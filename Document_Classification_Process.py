#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[1]:


import glob
import copy
import nltk
import scipy
import multiprocessing
import operator, os, pickle
import imp
import re
import joblib

import numpy as np
import pandas as pd
import utils as my_utils

from nltk.corpus import stopwords
from bs4 import BeautifulSoup as bs
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

from tqdm import tqdm
from collections import Counter
from itertools import combinations
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.model_selection import train_test_split


# # Necessary Functions

# In[2]:


stop_words = stopwords.words('english')
lemmatizer = nltk.stem.WordNetLemmatizer()
stemmer = nltk.stem.PorterStemmer()
w_tokenizer = nltk.tokenize.WhitespaceTokenizer()


# In[ ]:





# In[3]:


ner = set(pd.read_csv("ner_list", header=None, sep="\\").values.reshape(-1))
ner = {i:idx for idx, i in enumerate(ner)}


# In[4]:


def get_pos_tags(text):
    pos = nltk.pos_tag(text)

    m = {i:0 for i in [j[1] for j in pos]}
    m.keys
    for _, j in pos:
        m[j] += 1
    n = [0] * len(ner)
    for k, v in m.items():
        n[ner[k]] = v
    return n


# In[5]:


def get_special_count(inp):
    text, raw_text = inp
    v_raw_tl = len(raw_text)
    v_proc_tl = len(text)
    v_raw_wc = len(word_tokenize(raw_text))
    v_proc_wc = len(word_tokenize(text))
    v_num = len(re.findall(r'([\d])', raw_text))
    v_uppercase = len(re.findall(r'([A-Z])', raw_text))
    v_lower = len(re.findall(r'([a-z])', raw_text))
    v_spl_chrs = len(re.findall(r'([^(a-zA-Z\d \n)])', raw_text))
    v_new_line = len(re.findall(r'([\n])', raw_text))
    return [v_raw_tl, v_proc_tl, v_raw_wc, v_proc_wc, v_num, v_uppercase, v_lower, v_spl_chrs, v_new_line]


# In[6]:


def preprocessing_new(text):
    text = [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(re.sub(r'[^a-zA-Z]', " ", text.lower()))]
    return " ".join([i for i in text if len(i)>1])


# In[7]:


def get_vec(raw_text):
    text = preprocessing_new(raw_text)
    te_raw_pos = get_pos_tags(raw_text)
    te_spl_count = get_special_count([text, raw_text])
    te_vec = vectorizer.transform([text]).toarray()[0].tolist()
    return  te_raw_pos + te_spl_count + te_vec


# In[8]:


def predict(raw_text):
    ret = {}
    for i, j in zip(clf.classes_, clf.predict_proba([get_vec(raw_text)])[0]):
        ret[i] = j
    return ret


# In[ ]:





# In[9]:


vectorizer = joblib.load("vectorizer")


# In[10]:


clf = joblib.load("clf")


# In[ ]:





# In[11]:


raw_text = "Apple sells several different iPhone models — here’s how much they all cost Just about anyone living on planet Earth knows when to expect an onslaught of new iPhones released every year: September. For the past few years, Apple has made its annual iPhone lineup more complex by releasing multiple version of the iPhone each year.\n\nWe’ve tracked down prices from around the internet to answer the question, „How much does the iPhone cost?“ We break down the prices for all the models you can buy online from Apple, Walmart, Best Buy, and the four major cellular providers – Sprint, T-Mobile, Verizon, and AT&T – as well as the price on Amazon with Simple Mobile Prepaid.\n\nBefore getting into the details, the prices we’re listing are based on the lowest possible storage amount (64GB for the iPhone 8 to iPhone 11, 32GB for the iPhone 7, and 16GB for the iPhone 6S). If you want more storage, the price will be higher.\n\nAlso, as we get to the models that are a few years old, there may not be an option to buy it new. You can also read our full iPhone buying guide for more buying recommendations and advice.\n\nUpdated on 01/29/2020 by Joe Osborne: Updated prices, facts and language for the new year.\n\niPhone 11 price\n\nFoto: sourceCrystal Cox/Business Insider\n\nThe $699.99 iPhone 11 is the cheapest of the new iPhones and also the best value for the money, sporting a 6.1-inch LCD screen, Apple’s top-end A13 Bionic chip, a fantastic dual-camera setup, and over a full day of battery life.\n\n* Prices are for the 64GB model. If you want more storage, you’ll pay more.\n\niPhone 11 Pro price\n\nFoto: sourceCrystal Cox/Business Insider\n\nThe $999.99 iPhone 11 Pro has a smaller 5.8-inch OLED screen than the iPhone 11, as well as a stainless steel body and a three-camera setup that includes a telephoto lens, a wide-angle lens, and an ultra-wide lens.\n\n* Prices are for the 64GB model. If you want more storage, you’ll pay more.\n\niPhone 11 Pro Max price\n\nFoto: sourceCrystal Cox/Business Insider\n\nThe $1,099.99 iPhone 11 Pro Max is most expensive device in the iPhone 11 lineup. Compared to the iPhone 11 Pro, it has a bigger 6.5-inch screen, slightly higher-resolution display, and a longer battery life.\n\n* Prices are for the 64GB model. If you want more storage, you’ll pay more.\n\niPhone XS price\n\nFoto: sourceApple\n\nAlthough it is now a year old, the $899.99 iPhone XS is the a solid small, high-end iPhone with its gorgeous 4.7-inch OLED screen, fast processor, dual cameras, and comfortable size.\n\n* Prices are for the 64GB model. If you want more storage, you’ll pay more.\n\niPhone XS Max price\n\nFoto: sourceHollis Johnson/Business Insider\n\nThe $999.99 iPhone XS Max has a large 6.5-inch OLED screen, fast processor, long battery life, and great cameras, making it ideal for people who love big phones, but don’t want to buy the newest model.\n\n* Prices are for the 64GB model. If you want more storage, you’ll pay more.\n\niPhone XR price\n\nFoto: sourceJustin Sullivan/Getty Images\n\nThe $599.99 iPhone XR is still a great deal for iPhone bargain shoppers, as it has most of the same specs as the iPhone XS.\n\n* Prices are for the 64GB model. If you want more storage, you’ll pay more.\n\niPhone X price\n\nFoto: sourceHollis Johnson\n\nThe iPhone X is now two years old, and it still costs a lot at $800 or $900, so you’d do better to buy an iPhone 11 or an XS.\n\n* Prices are for the 64GB model. If you want more storage, you’ll pay more.\n\niPhone 8 price\n\nFoto: sourceHollis Johnson/Business Insider\n\nThe two-year-old iPhone 8 is small enough to hold comfortably in one hand, and it only costs $449.99 unlocked now, but you’d do better to buy the XR, which is nearly the same price now, or the 11 if you can swing it.\n\n* Prices are for the 64GB model. If you want more storage, you’ll pay more.\n\niPhone 8 Plus price\n\nFoto: sourceHollis Johnson/Business Insider\n\nAlthough it’s two years old now, the $549.99 iPhone 8 Plus is an OK buy for people who want the dual-camera tech for less and don’t mind an older design. If you can, though, you should pay just a bit more for the 11.\n\n* Prices are for the 64GB model. If you want more storage, you’ll pay more.\n\niPhone 7 price\n\nFoto: sourceHollis Johnson/Business Insider\n\nThe iPhone 7 is still a decent phone three years after its launch, but you’d do better to buy a newer model.\n\n* Prices are for the 32GB model. If you want more storage, you’ll pay more.\n\niPhone 7 Plus price\n\nFoto: sourceCorey Protin\n\nEven though it’s three years old, the iPhone 7 Plus is still a good phone and it has dual cameras. However, you’d be better off with a newer model.\n\n* Prices are for the 32GB model. If you want more storage, you’ll pay more.\n\niPhone 6S price\n\nFoto: sourceApple\n\nAlthough the iPhone 6S looks new, it has four-year-old tech inside and it won’t last much longer, so don’t buy it – even though it’s less than $200.\n\n* Prices are for the 16GB model unless otherwise noted. If you want more storage, you’ll pay more.\n\niPhone 6S Plus price\n\nFoto: sourceAntonio Villas-Boas/Tech Insider\n\nEven though the iPhone 6S Plus is just $299.99, it’s not worth buying because the tech is already outdated and it won’t last you as long as a newer phone.\n\n* Prices are for the 32GB model. If you want more storage, you’ll pay more.\n\nHow should you buy your new iPhone?\n\nFoto: sourceApple\n\nTech geek? Join the Apple iPhone Upgrade Program. You’ll essentially rent your phone with monthly payments, and you can upgrade to a new one after 12 payments. If you do that, you end up paying half price for the phone and you get a new one every year without worrying about the cost. T-Mobile has a similar offer, in which you can upgrade as soon as you’ve paid off half of the phone’s entire balance.\n\nJoin the Apple iPhone Upgrade Program. You’ll essentially rent your phone with monthly payments, and you can upgrade to a new one after 12 payments. If you do that, you end up paying half price for the phone and you get a new one every year without worrying about the cost. T-Mobile has a similar offer, in which you can upgrade as soon as you’ve paid off half of the phone’s entire balance. Tech-savvy traveler? We recommend you buy your iPhone unlocked so you can pop in local SIM cards when you travel. To do this, buy from Apple, preferably through the iPhone Upgrade Program. Alternatively, you can buy from T-Mobile or Sprint because they have free international service in 100+ countries.\n\nWe recommend you buy your iPhone unlocked so you can pop in local SIM cards when you travel. To do this, buy from Apple, preferably through the iPhone Upgrade Program. Alternatively, you can buy from T-Mobile or Sprint because they have free international service in 100+ countries. Budget hunter? Check out all the carrier promotions and maybe wait a few months after the launch to buy your new iPhone. The new iPhone 11 is a good deal at $699.99, as it has most of the same tech as the new 11 Pro models. You can also get the $599.99 iPhone XR, which costs several hundred dollars less than the XS and Max, but has most of the same tech. You can get a two-year-old iPhone 8 or iPhone 8 Plus for a good price now, too, but we don’t recommend them. Do not buy the iPhone 7 or older, though, because the tech is old, it won’t last as long, and it is not a good investment.\n\nCheck out the best iPhone cases for every model\n\nFoto: sourceOtterBox\n\nWe’ve rounded up the best iPhone case companies so you can find an excellent case for your iPhone no matter the model number or your style.\n\nWhether you want a fancy leather case, a folio, a rugged case or a basic no-frills one that gets the job done – we have a pick for you. We also have advice on how to figure out which iPhone model you have and which cases will fit it here.\n\nHere are the best iPhone cases for every model:"


# In[12]:


predict(raw_text)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




