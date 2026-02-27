import streamlit as st
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Download stopwords (first time only)
nltk.download('stopwords')

# -------------------------------
# LOAD DATASET (ERROR-PROOF)
# -------------------------------
df = pd.read_csv("fake_news.csv")

# Fix column names (remove spaces, lowercase)
df.columns = df.columns.str.strip().str.lower()

# Ensure required columns exist
if "text" not in df.columns or "label" not in df.columns:
    st.error("❌ CSV must contain columns named 'text' and 'label'")
    st.stop()

# -------------------------------
# PREPROCESSING FUNCTION
# -------------------------------
def preprocess(text):
    text = str(text).lower()
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

df["clean_text"] = df["text"].apply(preprocess)

# -------------------------------
# TF-IDF VECTORIZATION
# -------------------------------
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["clean_text"])

# Labels
y = df["label"]

# -------------------------------
# TRAIN MODEL
# -------------------------------
model = MultinomialNB()
model.fit(X, y)

# ===============================
#        STREAMLIT UI
# ===============================

st.title("📰 Fake News Detection App")

user_input = st.text_area("Enter news text:")

# -------------------------------
# PREDICTION WITH CONFIDENCE
# -------------------------------
if st.button("Predict"):

    if user_input.strip() != "":

        # Preprocess input
        clean = preprocess(user_input)

        # Convert to vector
        vec = vectorizer.transform([clean])

        # Prediction
        prediction = model.predict(vec)[0]

        # Probability (confidence)
        probabilities = model.predict_proba(vec)[0]
        confidence = max(probabilities) * 100

        # Display result
        if prediction == "REAL":
            st.success(f"✅ This news appears REAL ({confidence:.2f}% confidence)")
        else:
            st.error(f"⚠️ This news appears FAKE ({confidence:.2f}% confidence)")

    else:
        st.warning("Please enter text")