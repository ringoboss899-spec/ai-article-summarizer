import streamlit as st
import requests
from bs4 import BeautifulSoup
from transformers import pipeline
from deep_translator import GoogleTranslator
from youtube_transcript_api import YouTubeTranscriptApi
import re

# Page setup
st.set_page_config(page_title="Smart AI Summarizer", page_icon="🤖", layout="centered")

st.title("🤖 Smart AI Article & Video Summarizer")
st.write("Paste any news article or YouTube link, give custom instructions, and get instant insights in any language!")

# Hugging Face Summarization Pipeline
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = load_summarizer()

# Helper Function: Extract YouTube Video ID
def get_youtube_id(url):
    regex = r"(?:v=|\/)([0-9A-Za-z_-]{11})"
    match = re.search(regex, url)
    return match.group(1) if match else None

# Helper Function: Scrape Web Text
def scrape_article(url):
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    paragraphs = soup.find_all('p')
    text = " ".join([p.get_text() for p in paragraphs])
    return text

# Helper Function: Extract YouTube Subtitles
def get_youtube_transcript(video_id):
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = " ".join([item['text'] for item in transcript_list])
        return transcript_text
    except Exception as e:
        return None

# User Inputs
url = st.text_input("🔗 Enter News Article or YouTube URL:")
custom_prompt = st.text_area(
    "💬 Custom Instructions / Aapko AI se kya poochna hai?",
    value="Is link ke main points bullet points mein likh kar do, batao ye mere liye kyun kaam ki hai (14 years student ke liye easy language mein), aur 3 tips do.",
    height=100
)

if st.button("🚀 Analyze & Summarize"):
    if url:
        with st.spinner("Processing text and generating custom response..."):
            extracted_text = ""
            
            # Check if URL is YouTube
            if "youtube.com" in url or "youtu.be" in url:
                video_id = get_youtube_id(url)
                if video_id:
                    extracted_text = get_youtube_transcript(video_id)
                    if not extracted_text:
                        st.error("YouTube video ke subtitles/transcript nahi mil sake. Aisi video use karein jis mein subtitles enable ho.")
                else:
                    st.error("Invalid YouTube URL!")
            else:
                # Scrape general web article
                extracted_text = scrape_article(url)

            if extracted_text:
                # Truncate text to avoid model context overflow
                truncated_text = extracted_text[:3000]
                
                # Generate base summary
                raw_summary = summarizer(truncated_text, max_length=150, min_length=40, do_sample=False)[0]['summary_text']
                
                # Store in session state for translation
                st.session_state['summary'] = raw_summary
                st.session_state['custom_instructions'] = custom_prompt
                st.session_state['processed'] = True
    else:
        st.warning("Please enter a valid URL first!")

# Display Results & Urdu Translation
if st.session_state.get('processed'):
    st.subheader("📌 Key Summary:")
    st.write(st.session_state['summary'])

    st.subheader("💡 Your Custom Instructions:")
    st.info(st.session_state['custom_instructions'])

    # Urdu Translation Feature
    if st.button("🌐 Translate Summary to Urdu"):
        with st.spinner("Translating to Urdu..."):
            translated_text = GoogleTranslator(source='auto', target='ur').translate(st.session_state['summary'])
            st.success("**Urdu Translation:**")
            st.write(translated_text)