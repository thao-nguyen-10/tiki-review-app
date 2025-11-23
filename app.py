import streamlit as st
import pandas as pd
import joblib
import re
import os
from src.clean_text import clean_text

st.title("Product Review Multi-Aspect Classification Demo")

# ===============================
# 1. LOAD ALL MODELS
# ===============================

tfidf = joblib.load("trained_models/tfidf_vectorizer.pkl")

spam_clf = joblib.load("trained_models/spam_clf.pkl")
spam_encoder = joblib.load("trained_models/spam_encoder.pkl")

frequency_clf = joblib.load("trained_models/frequency_clf.pkl")
frequency_encoder = joblib.load("trained_models/frequency_encoder.pkl")

origin_clf = joblib.load("trained_models/origin_clf.pkl")
origin_encoder = joblib.load("trained_models/origin_encoder.pkl")

price_clf = joblib.load("trained_models/price_clf.pkl")
price_encoder = joblib.load("trained_models/price_encoder.pkl")

quality_clf = joblib.load("trained_models/quality_clf.pkl")
quality_encoder = joblib.load("trained_models/quality_encoder.pkl")

service_clf = joblib.load("trained_models/service_clf.pkl")
service_encoder = joblib.load("trained_models/service_encoder.pkl")

# ===============================
# 2. LOAD CRAWLED CSV AUTOMATICALLY
# ===============================

CRAWLED_CSV_PATH = "updated_reviews.csv"  # path where GitHub Action or your crawler saves the CSV

if not os.path.exists(CRAWLED_CSV_PATH):
    st.error(f"CSV file not found at {CRAWLED_CSV_PATH}. Make sure your crawler saved it.")
    st.stop()

df = pd.read_csv(CRAWLED_CSV_PATH)

if "content" not in df.columns:
    st.error("CSV must contain a 'content' column.")
    st.stop()

st.write("### Sample of crawled reviews:")
st.write(df.head())


# ===============================
# 4. CLEAN TEXT + TF-IDF TRANSFORM
# ===============================

df["clean_text"] = df["content"].astype(str).apply(clean_text)
X = tfidf.transform(df["clean_text"])


# ===============================
# 5. PREDICT ALL ASPECTS
# ===============================

df["spam"] = spam_encoder.inverse_transform(spam_clf.predict(X))

df["sent_frequency"] = frequency_encoder.inverse_transform(frequency_clf.predict(X))
df["sent_origin"] = origin_encoder.inverse_transform(origin_clf.predict(X))
df["sent_price"] = price_encoder.inverse_transform(price_clf.predict(X))
df["sent_quality"] = quality_encoder.inverse_transform(quality_clf.predict(X))
df["sent_service"] = service_encoder.inverse_transform(service_clf.predict(X))


# ===============================
# 6. SHOW RESULTS
# ===============================
st.write("### Predictions for latest reviews:")
st.write(df[[
    "content",
    "spam",
    "sent_frequency",
    "sent_origin",
    "sent_price",
    "sent_quality",
    "sent_service"
]].tail(50))


# ===============================
# 7. EXPORT CSV
# ===============================

st.download_button(
    label="Download Predicted CSV",
    data=df.to_csv(index=False),
    file_name="predicted_reviews.csv",
    mime="text/csv"
)