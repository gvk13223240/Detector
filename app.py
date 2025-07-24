# unified_app.py
import streamlit as st
import joblib
import numpy as np

# Set up app page
st.set_page_config(page_title="Mental Health Detector", layout="centered")
st.title("🧠 Mental Health Classification App")
st.markdown("Developed by **Garlapati Vamshi Krishna**\n\n🔍 _Select classification type and input text to predict mental health condition._")

# --- Step 1: User selects classification type ---
classification_type = st.radio(
    "Select Classification Type:",
    ["Binary Classification (Depressed / Not Depressed)", "Multi-Class Classification"]
)

# --- Step 2: Load appropriate models and vectorizers ---
@st.cache(allow_output_mutation=True)
def load_models(task):
    if task == "binary":
        return {
            "Logistic Regression": joblib.load("logistic_model.pkl"),
            "Naive Bayes": joblib.load("naive_bayes_model.pkl"),
            "Random Forest": joblib.load("random_model.pkl"),
            "SVM": joblib.load("svm_model.pkl"),
            "Voting Classifier": joblib.load("voting_model.pkl"),
        }, joblib.load("tfidf_vectorizer.pkl")
    else:
        return {
            "Logistic Regression": joblib.load("multi_logistic_model.pkl"),
            "Naive Bayes": joblib.load("multi_naive_bayes_model.pkl"),
            "Random Forest": joblib.load("multi_random_model.pkl"),
            "SVM": joblib.load("multi_svm_model.pkl"),
        }, joblib.load("multi_tfidf_vectorizer.pkl")

# Determine classification mode
is_binary = "Binary" in classification_type.lower()
models, vectorizer = load_models("binary" if is_binary else "multi")

# --- Step 3: User inputs text ---
user_input = st.text_area("Enter a social media post or text (Minimum 30 words):", height=150)

# --- Step 4: Predict ---
if st.button("Predict"):
    if len(user_input.strip().split()) < 30:
        st.warning("Please enter at least 30 words for reliable prediction.")
    else:
        X_input = vectorizer.transform([user_input])
        
        st.markdown("### 🔍 Model Predictions:")

        for name, model in models.items():
            prediction = model.predict(X_input)[0]

            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_input)[0]
                confidence = np.max(proba) * 100
                class_confidences = dict(zip(model.classes_, proba))
            else:
                confidence = None
                class_confidences = {}

            st.markdown(f"#### 🔹 {name}")
            st.success(f"**Predicted Class:** {prediction}")

            if confidence:
                st.info(f"Confidence: {confidence:.2f}%")

            if class_confidences:
                st.markdown("Class Probabilities:")
                sorted_probs = dict(sorted(class_confidences.items(), key=lambda x: x[1], reverse=True))
                for cls, prob in sorted_probs.items():
                    st.write(f"{cls}: **{prob * 100:.2f}%**")

st.markdown("---")
st.markdown("_This tool is for academic use only. Not a substitute for professional diagnosis._")
