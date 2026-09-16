import streamlit as st
import requests
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from textblob import TextBlob

# Page Configuration
st.set_page_config(page_title="Smart AI Summarizer", page_icon="📰", layout="wide")

st.title("📰 Smart AI Article Summarizer & Sentiment Analyzer")
st.write("Paste any news or article link below to get a concise summary and emotional tone analysis.")

# Direct Model Loading to avoid pipeline KeyError
@st.cache_resource
def load_model_and_tokenizer():
    model_name = "facebook/bart-large-cnn"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model_and_tokenizer()

# Scraping Function
def scrape_article(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        paragraphs = soup.find_all('p')
        text = ' '.join([p.text for p in paragraphs])
        return text
    except Exception:
        return None

# User Input Interface
url_input = st.text_input("🔗 Enter Article URL:", placeholder="https://www.bbc.com/news/article-example")

if st.button("🚀 Summarize & Analyze"):
    if url_input:
        with st.spinner("Fetching article content and generating summary... Please wait!"):
            article_text = scrape_article(url_input)
            
            if article_text and len(article_text.strip()) > 200:
                # 1. Summary Generation
                inputs = tokenizer([article_text[:3000]], max_length=1024, return_tensors="pt", truncation=True)
                summary_ids = model.generate(inputs["input_ids"], max_length=130, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
                summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                
                # 2. Sentiment Analysis
                blob = TextBlob(article_text)
                polarity = blob.sentiment.polarity
                
                if polarity > 0.1:
                    sentiment = "Positive 😊"
                elif polarity < -0.1:
                    sentiment = "Negative 😞"
                else:
                    sentiment = "Neutral 😐"

                # Display Results
                st.markdown("---")
                st.subheader("📌 Key Summary:")
                st.success(summary_text)
                
                st.subheader("📊 Tone & Sentiment:")
                st.info(f"Article Tone: **{sentiment}**")
            else:
                st.error("Could not extract enough text from this URL. Please try another news link.")
    else:
        st.warning("Please enter a valid URL.")