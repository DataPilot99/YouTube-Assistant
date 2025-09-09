import streamlit as st
import requests
import re

st.title('YouTube Assistant')

# Select language preference
with st.sidebar:
    language = st.radio('The bot can handle videos in English and Hindi.'
    'For videos in Hindi, please select language preference for the answer',
    ['Answer in original language', 'Translate to English'],
    index=0)


if 'messages' not in st.session_state:
    st.session_state.messages=[]

# Validate YouTube URL pattern
youtube_pattern = re.compile(
    r"^(https?://)?(www\.)?(youtube\.com|youtu\.be)/(watch\?v=[\w-]{11}|embed/[\w-]{11}|shorts/[\w-]{11}|[\w-]{11})"
)

link = st.sidebar.text_input('Paste YouTube video link here')

# ✅ Show validation error below input if invalid
if link and not youtube_pattern.match(link):
    st.sidebar.error("Please enter a valid YouTube URL")
elif not link:
    st.sidebar.warning("Please enter a YouTube URL")

prompt = st.chat_input('Ask a question from the video...')

if prompt:
    if not link or not youtube_pattern.match(link):
        # Block asking questions if URL is invalid
        st.session_state.messages.append({'role':'assistant', 'content': "Please enter a valid YouTube link before asking a question."})
    else:
        st.session_state.messages.append({'role':'user', 'content':prompt}) # Adding message to memory

        try:
            response = requests.post(url='http://127.0.0.1:1111/chat', json={'url':link, 'question':prompt,
                                                                             'language':language })
            data = response.json()
            answer = data.get("answer", "No answer returned.")
            
        except requests.exceptions.RequestException as e:
            answer = str(e)
        
        st.session_state.messages.append({'role':'assistant', 'content':answer}) # Adding answer to memory

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
