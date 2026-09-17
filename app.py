import streamlit as st
import requests
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from deep_translator import GoogleTranslator
from youtube_transcript_api import YouTubeTranscriptApi
import re

# Page Setup
st.set_page_config(page_title="Smart AI Summarizer", page_icon="🤖", layout="centered")

st.title("🤖 Smart AI Article & Video Summarizer")
st.write("Paste any news article or YouTube link, give custom instructions, and get instant insights!")

# Load Model
@st.cache_resource
def load_model_and_tokenizer():
    model_name = "sshleifer/distilbart-cnn-12-6"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model_and_tokenizer()

def generate_base_summary(text):
    inputs = tokenizer([text], max_length=1024, return_tensors="pt", truncation=True)
    summary_ids = model.generate(inputs["input_ids"], max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def get_youtube_id(url):
    pattern = r"(?:v=|\/|embed\/|youtu\.be\/)([0-9A-Za-z_-]{11})"
    match = re.search(pattern, url)
    return match.group(1) if match else None

def scrape_article(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        return " ".join([p.get_text() for p in paragraphs])
    except Exception:
        return None

def get_youtube_transcript(video_id):
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['en', 'en-US', 'ur', 'hi'])
        return " ".join([item['text'] for item in transcript_list])
    except Exception:
        return None

# User Inputs
url = st.text_input("🔗 Enter News Article or YouTube Link:")
manual_text = st.text_area("📝 Alternative: Paste Article / Video Transcript here directly (If link fails):", height=100)

custom_prompt = st.text_area(
    "💬 Custom Instructions / Prompt (Ask AI in any language):",
    value="Is content ke main points bullet points me likh kr do, ye mere liye kyn kaam ka hai, aur 3 tips do.",
    height=100
)

if st.button("🚀 Analyze & Summarize"):
    extracted_text = ""
    
    if manual_text.strip():
        extracted_text = manual_text
    elif url:
        with st.spinner("Processing link..."):
            if "youtube.com" in url or "youtu.be" in url:
                video_id = get_youtube_id(url)
                if video_id:
                    extracted_text = get_youtube_transcript(video_id)
                    if not extracted_text:
                        st.error("YouTube IP blocked this video. Please copy-paste video transcript/text in the box above!")
                else:
                    st.error("Invalid YouTube URL!")
            else:
                extracted_text = scrape_article(url)

    if extracted_text:
        with st.spinner("Generating summary..."):
            raw_summary = generate_base_summary(extracted_text)
            st.session_state['summary'] = raw_summary
            st.session_state['custom_prompt'] = custom_prompt
            st.session_state['processed'] = True
    else:
        st.warning("Please provide a URL or paste text directly.")

# Display Results & Safe Translation
if st.session_state.get('processed'):
    st.subheader("📌 Key Summary (English):")
    st.write(st.session_state['summary'])

    st.subheader("💡 Your Custom Prompt / Instructions:")
    st.info(st.session_state['custom_prompt'])

    if st.button("🌐 Safe Translate to Urdu / Roman Urdu"):
        with st.spinner("Translating safely..."):
            try:
                translated_text = GoogleTranslator(source='auto', target='ur').translate(st.session_state['summary'])
                st.success("Urdu Translation:")
                st.write(translated_text)
            except Exception as e:
                st.error("Translation rate limit reached. Please try clicking the translate button again in 30 seconds.")