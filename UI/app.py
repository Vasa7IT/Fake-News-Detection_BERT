import streamlit as st
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

# Load Model and Tokenizer from Hugging Face Hub
MODEL_PATH = "Balavasan/bert-fake-news-detector"  

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=2)

    st.success("Model loaded successfully from Hugging Face! ✅")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Streamlit App
st.title("📰 Fake News Detector")

# User Input
user_input = st.text_area("Enter news text:", "")

if st.button("Analyze"):
    if user_input.strip():
        # Tokenize Input
        inputs = tokenizer(user_input, padding="max_length", truncation=True, max_length=512, return_tensors="pt")

        # Get Model Predictions
        with torch.no_grad():
            logits = model(**inputs).logits

        # Apply Softmax to get Probabilities
        probs = F.softmax(logits, dim=1).squeeze().numpy()
        fake_prob, real_prob = probs[0], probs[1]

        # Display Results
        st.write(f"**Fake News Probability:** {fake_prob * 100:.2f}%")
        st.write(f"**Real News Probability:** {real_prob * 100:.2f}%")

        # Show Final Verdict
        result = "🟥 Fake News" if fake_prob > real_prob else "✅ Real News"
        st.subheader(f"Prediction: {result}")

    else:
        st.warning("Please enter some text to analyze.")
