import streamlit as st
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

nltk.download('stopwords', quiet=True)

st.set_page_config(page_title="Fake News Detection System", layout="wide")

# -------- LOAD DATA --------
df = pd.read_csv("fake_news.csv")
df.columns = df.columns.str.strip().str.lower()

def preprocess(text):
    text = str(text).lower()
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if w not in stop_words]
    return " ".join(tokens)

df["clean_text"] = df["text"].apply(preprocess)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["clean_text"])
y = df["label"]

model = MultinomialNB()
model.fit(X, y)

# -------- PAGE TITLE --------
st.title("📰 Fake News Detection System")
st.caption("AI-powered system for identifying misleading and unreliable information")

# -------- LAYOUT --------
left, center, right = st.columns([1, 2.5, 1])

# ================= LEFT PANEL =================
with left:
    st.subheader("📊 Dataset Overview")

    total = len(df)
    real_count = (df["label"] == "REAL").sum()
    fake_count = (df["label"] == "FAKE").sum()

    st.metric("Total Samples", total)
    st.metric("Real News", real_count)
    st.metric("Fake News", fake_count)

    st.markdown("---")

    st.subheader("⚙️ Model Info")
    st.write("Algorithm: Multinomial Naive Bayes")
    st.write("Features: TF-IDF")
    st.write("Task: Text Classification")

# ================= CENTER PANEL =================
with center:
    st.subheader("Enter Text to Analyze")

    user_input = st.text_area(
        "Paste news content or headline:",
        height=220,
        placeholder="Type or paste news text here..."
    )

    if st.button("🔍 Analyze News", use_container_width=True):

        if user_input.strip():

            clean = preprocess(user_input)
            vec = vectorizer.transform([clean])

            prediction = model.predict(vec)[0]
            probs = model.predict_proba(vec)[0]
            confidence = max(probs) * 100

            st.markdown("---")

            if prediction == "REAL":
                st.success(f"✅ REAL NEWS — Confidence: {confidence:.2f}%")
            else:
                st.error(f"🚨 FAKE NEWS — Confidence: {confidence:.2f}%")

            st.progress(int(confidence))

        else:
            st.warning("Please enter text.")

# ================= RIGHT PANEL =================
with right:
    st.subheader("ℹ️ About Fake News")

    st.info("""
**Fake news may include:**

• Fabricated stories  
• Conspiracy theories  
• Misleading headlines  
• Manipulated content  
• Unverified claims  
• Sensational rumors  
""")

    st.markdown("---")

    st.subheader("🌍 Impact")

    st.write("""
Fake news can influence public opinion,
spread panic, affect elections,
and cause social harm.
Reliable detection tools help
combat misinformation.
""")

# -------- FOOTER --------
st.markdown("---")
st.caption("Built with TF-IDF + Naive Bayes | Educational Project")
