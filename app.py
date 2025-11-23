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

# =========================
# Summary Statistics
# =========================
st.title("📊 Review Analytics Dashboard")

latest_time = pd.to_datetime(df['crawl_time']).max()
num_reviews = len(df)
num_customers = df['customer_id'].nunique() if 'customer_id' in df.columns else 0
num_sellers = df['seller_id'].nunique() if 'seller_id' in df.columns else 0

summary_df = pd.DataFrame({
    "Metric": [
        "Latest Crawl Time",
        "Number of Reviews",
        "Number of Customers",
        "Number of Sellers"
    ],
    "Value": [
        latest_time,
        num_reviews,
        num_customers,
        num_sellers
    ]
})

# ===== Custom column widths using HTML/CSS =====
st.subheader("📌 Summary")
st.markdown("""
<style>
.summary-table td:nth-child(1) { width: 200px !important; }
.summary-table td:nth-child(2) { width: 300px !important; }
</style>
""", unsafe_allow_html=True)

st.table(summary_df.style.set_table_attributes('class="summary-table"'))


# =========================
# Chart Grid: Class Distribution per Aspect
# =========================
st.subheader("📊 Class Distribution per Aspect")

aspects = ["spam_pred", "quality_pred", "service_pred", "origin_pred", "price_pred", "frequency_pred"]

n_cols = 3
# Loop through aspects in chunks of 3
for i in range(0, len(aspects), n_cols):
    row_aspects = aspects[i : i + n_cols]
    cols = st.columns(n_cols)

    for col, aspect in zip(cols, row_aspects):
        with col:
            fig, ax = plt.subplots()
            df[aspect].value_counts().plot(kind='bar', ax=ax)
            ax.set_title(f"Distribution of {aspect}")
            st.pyplot(fig)

# =========================
# Top Reviews per Class (2 × 2 tables)
# =========================
st.subheader("🌟 Top Reviews (<30 words) by Class")

# Function to get short reviews
def get_short_reviews(df, column, label, n=5):
    subset = df[(df[column] == label) & (df['content'].str.split().str.len() < 30)]
    
    # always return a list
    reviews = subset['content'].head(n).tolist()

    # pad to n rows
    if len(reviews) < n:
        reviews += [""] * (n - len(reviews))

    return reviews


for aspect in aspects:
    st.markdown(f"### 🔷 {aspect.replace('_pred','').title()}")

    # get unique classes
    aspect_classes = sorted(df[aspect].dropna().unique().tolist())

    table_dict = {}
    max_rows = 5

    for cls in aspect_classes:
        # ALWAYS returns a list of 5 items
        table_dict[cls.upper()] = get_short_reviews(df, aspect, cls, max_rows)

    # Build the table (now safe)
    aspect_table = pd.DataFrame.from_dict(table_dict)

    st.table(aspect_table)

# =========================
# WordCloud
# =========================
st.subheader("☁️ WordCloud of Customer Reviews")

text = " ".join(df['content'].astype(str).tolist())

wc = WordCloud(width=800, height=400).generate(text)

fig, ax = plt.subplots(figsize=(10, 5))
ax.imshow(wc, interpolation='bilinear')
ax.axis("off")
st.pyplot(fig)

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