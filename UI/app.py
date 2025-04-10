import streamlit as st
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
import requests
import re
from keybert import KeyBERT
import os
from dotenv import load_dotenv

# Load BERT Model
MODEL_PATH = "Balavasan/bert-fake-news-detector"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=2)

# App Title
st.title("📰 Fake News Detector")
st.markdown("---")

# Initialize session state
if "news_input" not in st.session_state:
    st.session_state.news_input = ""

# Text input box
user_input = st.text_area("Enter news text:", key="news_input")

# Analyze button
if st.button("Analyze"):
    if st.session_state.news_input.strip():
        inputs = tokenizer(
            st.session_state.news_input,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = F.softmax(logits, dim=1).squeeze().numpy()

        st.session_state.fake_prob = float(probs[0])
        st.session_state.real_prob = float(probs[1])
        st.session_state.result = "🟥 Fake News" if probs[0] > probs[1] else "✅ Real News"

# Show results (if available)
if "fake_prob" in st.session_state:
    st.write(f"**Fake News Probability:** {st.session_state.fake_prob * 100:.2f}%")
    st.write(f"**Real News Probability:** {st.session_state.real_prob * 100:.2f}%")
    st.subheader(f"Prediction: {st.session_state.result}")

# ----------------------------------
# 🔗 Suggest News Source Section
# ----------------------------------

st.markdown("---")
st.subheader("🔗 Possible News Source")

# Load environment variables from .env file
load_dotenv()

try:
    API_KEY = st.secrets["NEWS_API_KEY"]
except Exception:
    API_KEY = os.getenv("NEWS_API_KEY")
if not API_KEY:
    st.error("🚨 API key not found! Please set it in .env or secrets.toml")
    
kw_model = KeyBERT()

# Clean query utility
def get_clean_query(text):
    keywords = kw_model.extract_keywords(text, stop_words='english', top_n=5)
    return " ".join([kw[0] for kw in keywords])


def search_news_api(query, api_key):
    url = f"https://newsapi.org/v2/everything?q={query}&language=en&pageSize=3&apiKey={api_key}"
    try:
        response = requests.get(url)
        st.write("Response status:", response.status_code)
        if response.status_code == 200:
            return response.json().get("articles", [])
        else:
            st.error(f"NewsAPI error: {response.status_code}")
            return []
    except Exception as e:
        st.error(f"Request failed: {e}")
        return []


# Suggest Source button
if st.button("Suggest Source"):
    if st.session_state.news_input.strip():
        query = get_clean_query(st.session_state.news_input)
        st.write(f"Searching sources for: `{query}`")
        articles = search_news_api(get_clean_query(st.session_state.news_input), API_KEY)

        if articles:
            for art in articles:
                st.markdown(f"🔗 [{art['title']}]({art['url']})")
        else:
            st.info("No similar sources found.")
    else:
        st.warning("Please enter some news content first.")
