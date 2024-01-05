#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import streamlit as st 
from sklearn.neighbors import KNeighborsClassifier
from pickle import dump
from pickle import load


nltk.download('punkt')
nltk.download('stopwords')

clf = pickle.load(open('knn_mdl.sav', 'rb'))
st.title('Resume screening App : KNN Classifier ')


