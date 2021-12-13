# -*- coding: utf-8 -*-
#  File: streamlit_app.py
#  Project: 'ATOM test'
#  Created by Gennady Matveev (gm@og.ly) on 13-12-2021.
#  Copyright 2021. All rights reserved.
#  Inspiration/adopted from:
#  https://towardsdatascience.com/from-raw-data-to-web-app-deployment-with-atom-and-streamlit-d8df381aa19f 

#Import libraries
import pandas as pd
import streamlit as st
from atom import ATOMClassifier

# Init srteamlit
st.set_page_config(page_title='ATOM-ML demo',
                   page_icon='images/one.ico',
                   layout='wide',
                   initial_sidebar_state='expanded')

# Remove extra padding on the search web page
padding = 0
st.markdown(f""" <style>
    .reportview-container .main .block-container{{
        padding-top: {padding}rem;
        # padding-right: {padding}rem;
        # padding-left: {padding}rem;
        padding-bottom: {padding}rem;
    }} </style> """, unsafe_allow_html=True)

st.image('./images/header.png', use_column_width=True)

# Set up sidebar
st.sidebar.title("Pipeline")

# Data cleaning options
st.sidebar.subheader("Data cleaning")
scale = st.sidebar.checkbox("Scale", False, "scale")
encode = st.sidebar.checkbox("Encode", True, "encode")
impute = st.sidebar.checkbox("Impute", True, "impute")

# Model options
st.sidebar.subheader("Models")
models = {
    "gnb": st.sidebar.checkbox("Gaussian Naive Bayes", True, "gnb"),
    "rf": st.sidebar.checkbox("Random Forest", True, "rf"),
    "et": st.sidebar.checkbox("Extra-Trees", False, "et"),
    "xgb": st.sidebar.checkbox("XGBoost", False, "xgb"),
    "lgb": st.sidebar.checkbox("LightGBM", False, "lgb"),
    # "catb": st.sidebar.checkbox("CatBoost", False, "catb"),
}

# Page

st.header("Data")
st.write('Use provided dataset "weatheAUS.csv" or upload your own:')
waus = st. checkbox('weatherAUS')
if waus:
    data = './data/weatherAUS.csv'
else:
    data = st.file_uploader("Upload data:", type="csv")
# If a dataset is uploaded, show a preview
if data is not None:
    data = pd.read_csv(data)
    if st.checkbox('Show raw data'):
        st.write("Data preview:") #text
        st.dataframe(data.head())

st.header("Results")

if st.sidebar.button("Run"):
    if not (encode or impute):
        st.sidebar.warning('Please select more cleaning steps')
    else:
        placeholder = st.empty()  # Empty to overwrite write statements
        placeholder.write("Initializing atom...")
    
        # Initialize atom
        atom = ATOMClassifier(data, verbose=2, random_state=1)
    
        if scale:
            placeholder.write("Scaling the data...")
            atom.scale()
        if encode:
            placeholder.write("Encoding the categorical features...")
            atom.encode(strategy="LeaveOneOut", max_onehot=10)
        if impute:
            placeholder.write("Imputing the missing values...")
            atom.impute(strat_num="median", strat_cat="most_frequent")
        
        placeholder.write("Fitting the models...")
        to_run = [key for key, value in models.items() if value]
        atom.run(models=to_run, metric="f1")
        
        # Display metric results
        placeholder.write(atom.evaluate())
    
        # Draw plots
        col1, col2 = st.columns(2)
        col1.write(atom.plot_roc(title="ROC curve", display=None))
        col2.write(atom.plot_prc(title="PR curve", display=None))

else:
    st.write("No results yet. Click the run button!")
