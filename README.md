# AI-Powered YouTube Assistant (FastAPI + Streamlit)

This repository contains the code files to build a RAG based chatbot with additional web search feature. 
The chatbot can answer any question given a YouTube video link. If the answer to the user's question is not present in the video, the bot gives an answer from the web.

## Tech Stack
* LangChain
* Tavily API
* Fast API
* Streamlit
* OpenAI LLM
* Docker


## File Explanations

#### Backend (backend/)
* **api.py** → Defines the FastAPI app, routes, and endpoints. Handles incoming requests (e.g., from the Streamlit app) and calls the LLM pipeline.
* **pipeline.py** → Contains the RAG pipeline logic. This is where embeddings, retrieval, and model inference are defined.
* **requirements.txt** → Lists dependencies required to run the backend.
* **dockerfile** → Builds the backend Docker container.

#### Frontend (frontend/)
* **streamlit_app.py** → Defines the Streamlit UI. Provides an interface for users to input queries and displays results from the backend.
* **requirements.txt** → Lists dependencies required to run the frontend (e.g., Streamlit, requests, etc.).
* **dockerfile** → Builds the frontend Docker container. Installs dependencies and launches Streamlit.
