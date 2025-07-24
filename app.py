# app.py

import streamlit as st
import joblib
import numpy as np

# Mapping class index to labels
label_map = {0: "Not Depressed", 1: "Depressed"}

# Load all models and vectorizer
models = {
    "Logistic Regression": joblib.load("logistic_model.pkl"),
    "Naive Bayes": joblib.load("naive_bayes_model.pkl"),
    "Random Forest": joblib.load("random_model.pkl"),
    "SVM (Linear)": joblib.load("svm_model.pkl"),
    "Voting Classifier": joblib.load("voting_model.pkl"),
}
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Streamlit app configuration
st.set_page_config(page_title="Mental Health Classifier", layout="centered")
st.title("🧠 Mental Health Text Classifier")
st.markdown("This tool predicts whether a Reddit-style post indicates signs of depression.")

# Select model
model_name = st.selectbox("Select a model:", list(models.keys()))
model = models[model_name]

# Text input
user_input = st.text_area("✍️ Enter a mental health-related post (min 30 words):", height=180)

# Predict button
if st.button("🔍 Predict"):
    word_count = len(user_input.split())

    if word_count < 30:
        st.warning(f"Your input has only {word_count} words. Please enter at least 30 words.")
    elif user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Vectorize and predict
        X_input = vectorizer.transform([user_input])
        predicted_label = model.predict(X_input)[0]

        # Get probability/confidence
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_input)[0]
            confidence = np.max(proba) * 100
            class_confidences = dict(zip(model.classes_, proba))
        else:
            confidence = None
            class_confidences = {}

        # Display prediction result
        st.success(f"**Predicted Class:** {label_map[predicted_label]}")

        # Confidence level display
        if confidence:
            st.info(f"Confidence: {confidence:.2f}%")
            if predicted_label == 0 and confidence < 60:
                st.warning("⚠️ The model is not very confident. Although it predicts 'Not Depressed', the post might need further review.")

        # Show class probabilities
        if class_confidences:
            st.markdown("### 🔢 Class Probabilities")
            sorted_probs = dict(sorted(class_confidences.items(), key=lambda item: item[1], reverse=True))
            for cls, prob in sorted_probs.items():
                st.write(f"{label_map[cls]}: **{prob * 100:.2f}%**")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align:center;'>Work in progress by <strong>Garlapati Vamshi Krishna</strong> 🚀</div>",
    unsafe_allow_html=True
)
