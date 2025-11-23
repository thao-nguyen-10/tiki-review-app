import streamlit as st
import pandas as pd
import joblib
import re
import os
from src.clean_text import clean_text
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns

# -------------------------
# LOAD MODELS
# -------------------------
@st.cache_resource
def load_all_models():
    models = {
        "vectorizer": joblib.load("trained_models/tfidf_vectorizer.pkl"),

        "spam_clf": joblib.load("trained_models/spam_clf.pkl"),
        "spam_enc": joblib.load("trained_models/spam_encoder.pkl"),

        "freq_clf": joblib.load("trained_models/frequency_clf.pkl"),
        "freq_enc": joblib.load("trained_models/frequency_encoder.pkl"),

        "origin_clf": joblib.load("trained_models/origin_clf.pkl"),
        "origin_enc": joblib.load("trained_models/origin_encoder.pkl"),

        "price_clf": joblib.load("trained_models/price_clf.pkl"),
        "price_enc": joblib.load("trained_models/price_encoder.pkl"),

        "quality_clf": joblib.load("trained_models/quality_clf.pkl"),
        "quality_enc": joblib.load("trained_models/quality_encoder.pkl"),

        "service_clf": joblib.load("trained_models/service_clf.pkl"),
        "service_enc": joblib.load("trained_models/service_encoder.pkl"),
    }
    return models

models = load_all_models()

# -------------------------
# LOAD DATA
# -------------------------
DATA_PATH = "updated_reviews.csv"
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df
df = load_data()

# Preprocess + Predict all dataset
def predict_all(df):
    clean_texts = df["content"].fillna("").apply(clean_text)
    vectors = models["vectorizer"].transform(clean_texts)

    df["spam_pred"] = models["spam_enc"].inverse_transform(models["spam_clf"].predict(vectors))
    df["frequency_pred"] = models["freq_enc"].inverse_transform(models["freq_clf"].predict(vectors))
    df["origin_pred"] = models["origin_enc"].inverse_transform(models["origin_clf"].predict(vectors))
    df["price_pred"] = models["price_enc"].inverse_transform(models["price_clf"].predict(vectors))
    df["quality_pred"] = models["quality_enc"].inverse_transform(models["quality_clf"].predict(vectors))
    df["service_pred"] = models["service_enc"].inverse_transform(models["service_clf"].predict(vectors))

    return df

df = predict_all(df)

# -------------------------
#  UI START
# -------------------------
st.title("🛒 Product Review Analyzer — Auto Updated Dashboard")

st.markdown("This dashboard displays the latest crawled review data along with predictions.")

# -------------------------
# 1️⃣ SUMMARY GRID
# -------------------------
st.subheader("📊 Dataset Summary")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Last Updated", df["crawl_time"].max() if "crawl_time" in df.columns else "N/A")

with col2:
    st.metric("Total Reviews", len(df))

with col3:
    st.metric("Unique Customers", df["customer_id"].nunique())

with col4:
    st.metric("Unique Sellers", df["seller_id"].nunique())

# -------------------------
# 2️⃣ CHARTS
# -------------------------
st.subheader("📈 Class Distribution per Aspect")

aspects = {
    "spam_pred": "Spam Classification",
    "frequency_pred": "Shopping Frequency",
    "origin_pred": "Product Origin",
    "price_pred": "Price Sentiment",
    "quality_pred": "Product Quality",
    "service_pred": "Delivery Service"
}

for col, title in aspects.items():
    st.write(f"### 🔹 {title}")
    fig, ax = plt.subplots()
    sns.countplot(x=df[col], ax=ax)
    ax.set_title(f"Distribution of {title}")
    st.pyplot(fig)

# -------------------------
# 3️⃣ TOP 5 REVIEWS PER CLASS
# -------------------------
st.subheader("🏆 Top 5 Reviews per Class")

aspect_choice = st.selectbox("Choose aspect:", list(aspects.keys()))
classes = df[aspect_choice].unique()

for cls in classes:
    st.write(f"### 🔸 {cls}")
    top_reviews = df[df[aspect_choice] == cls].head(5)[["content"]]
    for i, row in top_reviews.iterrows():
        st.write(f"- {row['content']}")

# -------------------------
# 4️⃣ WORDCLOUD
# -------------------------
st.subheader("☁ WordCloud of All Reviews")

all_text = " ".join(df["content"].fillna("").astype(str).tolist())
wc = WordCloud(width=800, height=400, background_color="white").generate(all_text)

fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
ax_wc.imshow(wc, interpolation="bilinear")
ax_wc.axis("off")
st.pyplot(fig_wc)

# -------------------------
# Allow user to analyze one review
# -------------------------
st.subheader("🔍 Analyze a Single Review")

review = st.selectbox("Pick a review:", df["content"].dropna().tail(100))
cleaned = clean_text(review)
vector = models["vectorizer"].transform([cleaned])

st.write({
    "Spam": models["spam_enc"].inverse_transform(models["spam_clf"].predict(vector))[0],
    "Frequency": models["freq_enc"].inverse_transform(models["freq_clf"].predict(vector))[0],
    "Origin": models["origin_enc"].inverse_transform(models["origin_clf"].predict(vector))[0],
    "Price": models["price_enc"].inverse_transform(models["price_clf"].predict(vector))[0],
    "Quality": models["quality_enc"].inverse_transform(models["quality_clf"].predict(vector))[0],
    "Service": models["service_enc"].inverse_transform(models["service_clf"].predict(vector))[0],
})