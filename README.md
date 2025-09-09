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

## Getting Started
### Environment Setup
First, create a virtual environment and activate it:

```python -m venv venv
source venv/bin/activate   # On Mac/Linux
venv\Scripts\activate      # On Windows
```
Create a local environment file (.env) and add your API keys:
```
OPENAI_API_KEY=your_openai_api_key
```
### Install Dependencies
Install dependencies separately for backend and frontend:

Backend
```
cd backend
pip install -r requirements.txt
```

Frontend
```
cd ../frontend
pip install -r requirements.txt
```

### Run the Application
Start the Backend (FastAPI)
```
cd backend
uvicorn api:app --host 0.0.0.0 --port 8000
```


The API will be available at:
http://localhost:8000/docs

### Start the Frontend (Streamlit)

In a separate terminal:
```
cd frontend
streamlit run streamlit_app.py 
```


The frontend UI will be available at:
http://localhost:8501

### Usage

Once both services are running:

* Open the Streamlit frontend in your browser (http://localhost:8501).
* Enter your query or text in the interface.
* The frontend sends the request to the FastAPI backend, which processes it using pipeline.py.
* Results are displayed instantly in the UI.

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
