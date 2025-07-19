# app.py

import streamlit as st
import joblib
import numpy as np

# Load model and TF-IDF vectorizer
model = joblib.load("logistic_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# App title
st.title("🧠 Mental Health Sentiment Classifier")
st.write("This app analyzes text to predict whether it's likely written by someone experiencing depression.")

# Text input
user_input = st.text_area("📝 Enter a sentence or paragraph here:", height=150)

if st.button("🔍 Analyze"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text before analyzing.")
    else:
        # Vectorize input text
        X_input = vectorizer.transform([user_input])

        # Predict
        prediction = model.predict(X_input)[0]
        probability = model.predict_proba(X_input)[0][prediction] * 100

        # Result
        if prediction == 1:
            st.error(f"😟 Likely Depressed (Confidence: {probability:.2f}%)")
            st.info("📌 Note: This is not a medical diagnosis. If you're feeling down, talk to someone you trust or seek help.")
        else:
            st.success(f"😊 Not Depressed (Confidence: {probability:.2f}%)")
            st.info("👍 Keep staying positive and looking after your mental health!")
