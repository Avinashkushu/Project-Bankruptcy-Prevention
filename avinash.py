# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 13:01:17 2020

"""



import pandas as pd
import streamlit as st 
from sklearn.linear_model import LogisticRegression
from pickle import dump
from pickle import load

st.title('Model Deployment: Logistic Regression')

st.sidebar.header('User Input Parameters')

def user_input_features():
    industrial_risk = st.sidebar.selectbox('industrial_risk',('0','0.5','1'))
    management_risk = st.sidebar.selectbox('management_risk',('0','0.5','1'))
    financial_flexibility = st.sidebar.selectbox('financial_flexibility',('0','0.5','1'))
    credibility = st.sidebar.selectbox("credibility",('0','0.5','1'))
    competitiveness = st.sidebar.selectbox("competitiveness",('0','0.5','1'))
    operating_risk = st.sidebar.selectbox("operating_risk ",('0','0.5','1'))
    data = {'industrial_risk':industrial_risk,
            'management_risk':management_risk,
            'financial_flexibility':financial_flexibility,
            'credibility':credibility,
            'competitiveness':competitiveness,
            'operating_risk':operating_risk}
    features = pd.DataFrame(data,index = [0])
    return features 
    
df = user_input_features()
st.subheader('User Input parameters')
st.write(df)


# load the model from disk
loaded_model = load(open('logistic_mdl.sav', 'rb'))

prediction = loaded_model.predict(df)
prediction_proba = loaded_model.predict_proba(df)

st.subheader('Predicted Result')
st.write('Non-bankruptcy' if prediction_proba[0][1] > 0.5 else 'Bankruptcy')

st.subheader('Prediction Probability')
st.write(prediction_proba)


