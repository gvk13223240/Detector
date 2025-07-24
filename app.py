import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Mapping of class index to labels
label_map = {0: "Not Depressed", 1: "Depressed"}

# Load models
models = {
    "Logistic Regression": joblib.load("logistic_model.pkl"),
    "Naive Bayes": joblib.load("naive_bayes_model.pkl"),
    "Random Forest": joblib.load("random_model.pkl"),
    "SVM (Linear)": joblib.load("svm_model.pkl"),
    "Voting Classifier": joblib.load("voting_model.pkl"),
}
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Streamlit UI Setup
st.set_page_config(page_title="Mental Health Classifier", layout="centered")
st.title("🧠 Mental Health Text Classifier")
st.markdown("This app checks your input across **five different machine learning models** to detect signs of depression.")

# Text Input
user_input = st.text_area("✍️ Enter a mental health-related post (min 30 words):", height=180)

# Predict
if st.button("🔍 Predict with All Models"):
    word_count = len(user_input.split())

    if word_count < 30:
        st.warning(f"Your input has only {word_count} words. Please enter at least 30 words.")
    elif user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        X_input = vectorizer.transform([user_input])
        results = []

        for name, model in models.items():
            # Predict class
            prediction = model.predict(X_input)[0]

            # Get probabilities
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_input)[0]
                confidence = np.max(proba) * 100
                class_probs = dict(zip(model.classes_, proba))
            else:
                confidence = None
                class_probs = {}

            # Display each model’s result
            st.subheader(f"🔹 {name}")
            st.write(f"**Prediction:** {label_map[prediction]}")
            if confidence:
                st.write(f"**Confidence:** {confidence:.2f}%")
                if prediction == 0 and confidence < 60:
                    st.warning("⚠️ Low confidence: This post may still require further attention.")

            if class_probs:
                st.markdown("Class Probabilities:")
                for cls, prob in sorted(class_probs.items(), key=lambda x: x[1], reverse=True):
                    st.write(f"{label_map[cls]}: **{prob * 100:.2f}%**")

            # Append to summary
            results.append({
                "Model": name,
                "Predicted Class": label_map[prediction],
                "Confidence (%)": round(confidence, 2) if confidence else "N/A"
            })

        # Show comparison table
        st.markdown("---")
        st.subheader("📊 Model Comparison Summary")
        df = pd.DataFrame(results)
        st.dataframe(df)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align:center;'>Work in progress by <strong>Garlapati Vamshi Krishna</strong> 🚀</div>",
    unsafe_allow_html=True
)
