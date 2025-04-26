import streamlit as st
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
import requests
from keybert import KeyBERT
import os
from dotenv import load_dotenv
from PIL import Image
import pytesseract
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import login

login(token=os.getenv("HF_TOKEN"))

MODEL_PATH = "Balavasan/Fine_tuned-BERT"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_auth_token=True)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, use_auth_token=True)
# ------------------------
# local model path
# MODEL_PATH = r"E:\Productivity\git-Proj\main-proj\BERT-Fake_News_Detection\new-model-train"

# Load tokenizer and model from local path
# tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
# model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, local_files_only=True)


# ------------------------
# App Title
st.title("📰 Fake News Detector with Source Suggestion")
st.markdown("---")

# ------------------------
# Initialize session state
if "news_input" not in st.session_state:
    st.session_state.news_input = ""

# ------------------------
# Image upload and OCR extraction
uploaded_file = st.file_uploader("Or upload an image with news content", type=["png", "jpg", "jpeg"])
if uploaded_file is not None and st.button("Extract Text from Image"):
    try:
        image = Image.open(uploaded_file)
        extracted_text = pytesseract.image_to_string(image)
        st.session_state.news_input = extracted_text
        st.write("Extracted text from image:")
        st.info(extracted_text)
    except Exception as e:
        st.error(f"❌ OCR failed: {e}")

# ------------------------
# Text input box
user_input = st.text_area("Enter news text:", key="news_input", value=st.session_state.news_input)

# ------------------------
# Load environment variables from .env file
load_dotenv()
try:
    API_KEY = st.secrets["NEWS_API_KEY"]
except Exception:
    API_KEY = os.getenv("NEWS_API_KEY")

if not API_KEY:
    st.error("🚨 API key not found! Please set it in .env or secrets.toml")

# ------------------------
# KeyBERT for keyword extraction
kw_model = KeyBERT()

def get_clean_query(text):
    keywords = kw_model.extract_keywords(text, stop_words='english', top_n=5)
    return " ".join([kw[0] for kw in keywords])

def search_news_api(query, api_key):
    url = f"https://newsapi.org/v2/everything?q={query}&language=en&pageSize=3&apiKey={api_key}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.json().get("articles", [])
        else:
            st.error(f"NewsAPI error: {response.status_code}")
            return []
    except Exception as e:
        st.error(f"Request failed: {e}")
        return []

# ------------------------
# Analyze button logic
if st.button("Analyze"):
    input_text = user_input.strip()

    if input_text:
        # ---- Prediction ----
        inputs = tokenizer(
            input_text,
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

        threshold = 0.6
        if probs[0] > threshold:
            st.session_state.result = "🟥 Fake News"
        elif probs[1] > threshold:
            st.session_state.result = "✅ Real News"
        else:
            st.session_state.result = "🤔 Uncertain"

        # ---- Display prediction ----
        st.write(f"**Fake News Probability:** {st.session_state.fake_prob * 100:.2f}%")
        st.write(f"**Real News Probability:** {st.session_state.real_prob * 100:.2f}%")
        st.subheader(f"Prediction: {st.session_state.result}")

        # ---- Suggest similar sources ----
        st.markdown("---")
        st.subheader("🔗 Possible News Source")

        query = get_clean_query(input_text)
        st.write(f"Searching sources for: `{query}`")
        articles = search_news_api(query, API_KEY)

        if articles:
            for art in articles:
                st.markdown(f"🔗 [{art['title']}]({art['url']})")
        else:
            st.info("No similar sources found.")
    else:
        st.warning("Please enter some news content or upload an image first.")
