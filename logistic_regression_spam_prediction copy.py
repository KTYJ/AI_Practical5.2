import streamlit as st
from joblib import load
import numpy as np

#load le model
lr = load('logistic_regression_model.joblib')
cv = load('countvectorizer.joblib')

# Create a simple user input
#user_input = st.number_input('Enter house size:', min_value=100, max_value=10000, step=50)    <-- This is just example, ignore yea
user_input = st.text_input("Enter any sentence... :")


# Predict the house price
if st.button('Predict Spam/Ham'):
    Snew = cv.transform([user_input])
    result = lr.predict(Snew)
    st.write(f"The predicted sentiment is: {result[0]}")
