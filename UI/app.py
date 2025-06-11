import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from PIL import Image
import pytesseract
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv
from huggingface_hub import login

# Load environment variables
load_dotenv()

hf_token = os.getenv("HF_TOKEN")
api_key = os.getenv("NEWS_API_KEY")

# Hugging Face authentication
if hf_token:
    login(token=hf_token)
else:
    st.warning("Hugging Face token not found. Model loading may fail.")

# Model path
FAKE_NEWS_MODEL_PATH = "Balavasan/Fine_tuned-BERT"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(FAKE_NEWS_MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(FAKE_NEWS_MODEL_PATH)

# Load KeyBERT model safely with SentenceTransformer
try:
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    kw_model = KeyBERT(model=embedding_model)
except NotImplementedError:
    st.error("⚠️ SentenceTransformer failed to load due to cloud limitations. Keyword extraction won't work.")
    kw_model = None
# embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
# kw_model = KeyBERT(model=embedding_model)

# App UI
st.title("📰 Fake News Detector with Source Suggestion")
st.markdown("---")

if "news_input" not in st.session_state:
    st.session_state.news_input = ""

# Image upload and OCR
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

# Text input
user_input = st.text_area("Enter news text:", key="news_input", value=st.session_state.news_input)

def get_clean_query(text):
    if not kw_model:
        return ""
    keywords = kw_model.extract_keywords(text, stop_words='english', top_n=5)
    return " ".join([kw[0] for kw in keywords])


def search_news_api(query, api_key):
    import requests
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

# Analyze button
if st.button("Analyze"):
    input_text = user_input.strip()

    if not input_text:
        st.warning("Please enter some news content or upload an image first.")
    elif not api_key:
        st.error("🚨 NewsAPI key not found. Please check your .env file.")
    else:
        inputs = tokenizer(
            input_text,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.nn.functional.softmax(logits, dim=1).squeeze().numpy()

        st.session_state.fake_prob = float(probs[0])
        st.session_state.real_prob = float(probs[1])

        threshold = 0.6
        if probs[0] > threshold:
            st.session_state.result = "🟥 Fake News"
        elif probs[1] > threshold:
            st.session_state.result = "✅ Real News"
        else:
            st.session_state.result = "🤔 Uncertain"

        st.write(f"**Fake News Probability:** {st.session_state.fake_prob * 100:.2f}%")
        st.write(f"**Real News Probability:** {st.session_state.real_prob * 100:.2f}%")
        st.subheader(f"Prediction: {st.session_state.result}")

        # Source suggestion
        st.markdown("---")
        st.subheader("🔗 Possible News Source")

        query = get_clean_query(input_text)
        st.write(f"Searching sources for: `{query}`")
        articles = search_news_api(query, api_key)

        if articles:
            for art in articles:
                st.markdown(f"🔗 [{art['title']}]({art['url']})")
        else:
            st.info("No similar sources found.")
