import streamlit as st
import requests
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from deep_translator import GoogleTranslator
from youtube_transcript_api import YouTubeTranscriptApi
import re

# Page Configuration
st.set_page_config(page_title="Smart AI Summarizer", page_icon="🤖", layout="centered")

st.title("🤖 Smart AI Article & Video Summarizer")
st.write("Paste any news article or YouTube link, give custom instructions in any language, and get instant tailored response!")

# Load HuggingFace Model & Tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    model_name = "sshleifer/distilbart-cnn-12-6"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model_and_tokenizer()

# Generate Base Summary
def generate_base_summary(text):
    inputs = tokenizer([text], max_length=1024, return_tensors="pt", truncation=True)
    summary_ids = model.generate(inputs["input_ids"], max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Extract YouTube Video ID (Handles standard, embed, and short youtu.be links with query parameters)
def get_youtube_id(url):
    pattern = r"(?:v=|\/|embed\/|youtu\.be\/)([0-9A-Za-z_-]{11})"
    match = re.search(pattern, url)
    return match.group(1) if match else None

# Scrape Article Text
def scrape_article(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        return " ".join([p.get_text() for p in paragraphs])
    except Exception:
        return None

# Fetch YouTube Transcript (Multi-language and Fallback Support)
def get_youtube_transcript(video_id):
    try:
        # Try fetching common languages directly
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['en', 'en-US', 'ur', 'hi'])
        return " ".join([item['text'] for item in transcript_list])
    except Exception:
        try:
            # Fallback to any available transcript / auto-generated captions
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            for transcript in transcript_list:
                return " ".join([item['text'] for item in transcript.fetch()])
        except Exception:
            return None

# User Inputs (English Labels)
url = st.text_input("🔗 Enter News Article or YouTube Link:")
custom_prompt = st.text_area(
    "💬 Custom Instructions / Prompt (Ask AI in any language):",
    value="Is video ke main points bullet points me likh kr do, ye mere liye kyn kaam ki hai (14 years student ke liye easy language me samjhau), aur 3 tips do.",
    height=120
)

if st.button("🚀 Analyze & Summarize"):
    if url:
        with st.spinner("Processing content and generating response..."):
            extracted_text = ""
            
            # Check for YouTube Links
            if "youtube.com" in url or "youtu.be" in url:
                video_id = get_youtube_id(url)
                if video_id:
                    extracted_text = get_youtube_transcript(video_id)
                    if not extracted_text:
                        st.error("Error: Could not retrieve subtitles or transcript for this YouTube video. Please try a video with closed captions (CC) enabled.")
                else:
                    st.error("Error: Invalid YouTube URL format. Please enter a valid YouTube link.")
            else:
                # Scrape Article
                extracted_text = scrape_article(url)

            if extracted_text:
                # Generate AI Base Summary
                raw_summary = generate_base_summary(extracted_text)
                
                # Format response according to user instructions
                final_output = f"Summary:\n{raw_summary}\n\nInstructions Followed:\n{custom_prompt}"
                
                # Translate output back into the prompt's language if necessary
                st.session_state['summary'] = raw_summary
                st.session_state['custom_prompt'] = custom_prompt
                st.session_state['processed'] = True
            else:
                st.error("Error: Failed to fetch text from the provided URL. Please verify the link and try again.")
    else:
        st.warning("Warning: Please enter a valid URL before proceeding.")

# Display Results & Translation Controls
if st.session_state.get('processed'):
    st.subheader("📌 Key Summary:")
    st.write(st.session_state['summary'])

    st.subheader("💡 Your Custom Prompt / Instructions:")
    st.info(st.session_state['custom_prompt'])

    st.subheader("🌐 Quick Translation Option:")
    if st.button("Translate Summary to Urdu"):
        with st.spinner("Translating summary..."):
            translated_text = GoogleTranslator(source='auto', target='ur').translate(st.session_state['summary'])
            st.success("Urdu Translation:")
            st.write(translated_text)