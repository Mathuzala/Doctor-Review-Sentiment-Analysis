import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import re

#Function to preprocess the text
def preprocess_text(text):
    text = text.lower()  #Converting to lower case
    text = re.sub('[^a-zA-Z]', ' ', text)  #Removing punctuation and numbers
    text = re.sub(r'\s+', ' ', text)  #Removing multiple spaces
    return text

#Load your trained model and TF-IDF vectorizer
#For demonstration, replace these with the paths to your saved models
model = joblib.load('your_custom_path\\logistic_regression_model.pkl')
tfidf_vectorizer = joblib.load('your_custom_path\\tfidf_vectorizer.pkl')

#Streamlit application
st.title('Doctor Review Sentiment Analysis')
st.write('Enter a doctor review to predict the sentiment:')

#Text input from user
user_input = st.text_area("Review Text", "")

#Predicting Sentiment
if st.button('Predict Sentiment'):
    #Preprocessing the text
    processed_text = preprocess_text(user_input)
    #Vectorizing the text
    vectorized_text = tfidf_vectorizer.transform([processed_text])
    #Predicting the sentiment
    prediction = model.predict(vectorized_text)
    sentiment = 'Positive' if prediction == 1 else 'Negative'
    st.write(f'Predicted Sentiment: {sentiment}')

#Run the app with - streamlit run app.py
