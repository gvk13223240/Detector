# app.py
import streamlit as st
import joblib
import numpy as np

# Load all models and vectorizer
models = {
    "Logistic Regression": joblib.load("logistic_model.pkl"),
    "Naive Bayes": joblib.load("naive_bayes_model.pkl"),
    "Random Forest": joblib.load("random_model.pkl"),
    "SVM (Linear)": joblib.load("svm_model.pkl"),
    "Voting Classifier": joblib.load("voting_model.pkl"),
}
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Page config
st.set_page_config(page_title="Mental Health Classifier", layout="centered")

# Title and credits
st.title("Mental Health Text Classifier")
st.markdown("**Author:** Garlapati Vamshi Krishna")
st.markdown("**Status:** *Work in Progress*")
st.markdown("Enter a mental health-related Reddit-style post. Minimum 30 words. Select a model and get predictions.")

# Model selection
selected_model_name = st.selectbox("Choose a model:", list(models.keys()))
model = models[selected_model_name]

# Text input
user_input = st.text_area("Enter a post here:", height=150)

# Predict
if st.button("Predict"):
    word_count = len(user_input.strip().split())

    if user_input.strip() == "":
        st.warning("Please enter some text.")
    elif word_count < 30:
        st.warning(f"Minimum 30 words required. You entered {word_count}.")
    else:
        # Vectorize and predict
        X_input = vectorizer.transform([user_input])
        prediction = model.predict(X_input)[0]

        # Handle probabilities (if available)
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_input)[0]
            confidence = np.max(proba) * 100
            class_confidences = dict(zip(model.classes_, proba))
        else:
            confidence = None
            class_confidences = {}

        # Show result
        st.success(f"**Predicted Class:** {prediction}")
        if confidence:
            st.info(f"Confidence: {confidence:.2f}%")

        # Show class probabilities
        if class_confidences:
            st.markdown("Class Probabilities")
            sorted_probs = dict(sorted(class_confidences.items(), key=lambda item: item[1], reverse=True))
            for cls, prob in sorted_probs.items():
                st.write(f"- **{cls}**: {prob * 100:.2f}%")
