import os
from dotenv import load_dotenv
import re

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled

from langchain_tavily import TavilySearch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

from collections import defaultdict
from langchain.memory import ConversationSummaryMemory


# -------------------------------------   SETUP   ---------------------------------------------------------

load_dotenv()

llm_api_key = os.getenv('OPENAI_API_KEY')
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.2, api_key=llm_api_key)

parser = StrOutputParser()

# Dictionary to hold memory objects per video
video_memories = defaultdict(
    lambda: ConversationSummaryMemory(llm=llm, return_messages=True)
)



#--------------------------------  RETRIEVAL AUGMENTED GENERATION  -----------------------------------------

# ------HELPER FUNCTIONS--------


def get_memory_summary(yt_link: str) -> str:
    video_id = extract_youtube_id(yt_link)
    memory = video_memories[video_id]    
    return memory.load_memory_variables({}).get("history", "")


# Extracting video transcript from yt_link
def extract_youtube_id(url):
    # Regex to capture the video ID from different YouTube URL formats
    regex = r"(?:https?:\/\/)?(?:www\.)?(?:m\.)?(?:youtube\.com|youtu\.be)\/(?:watch\?v=|embed\/|v\/|e\/|shorts\/|user\/.+\/|ytscreeningroom\?v=|\/c\/.+\/videos?|\/channel\/.+\/videos?|)([\w-]{11})(?:\S+)?"
    match = re.search(regex, url)
    error_msg = 'The given link is not a valid YouTube url.'
    if match:
        extracted_video_id = match.group(1)
        return extracted_video_id
    return error_msg

def fetch_transcript(video_id):
    try:
        # fetches captions with timestamps
        yt_api = YouTubeTranscriptApi()
        transcript_list = yt_api.fetch(video_id, languages=['en', 'hi'])
        # Join the text piece of every timestamp
        return " ".join(chunk.text for chunk in transcript_list)
    except TranscriptsDisabled:
        return ''

def split_store_retrieve(transcript):
    # Splitting the transcript into chunks of 1000. create_documents converts raw strings 
    # to LangChain Document objects
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript])

    # Embedding generation and storing in ChromaDB
    embeddings = OpenAIEmbeddings(model='text-embedding-3-large',
                              api_key=llm_api_key)
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name='video-transcripts'
    )
    # 'retriever' is a Runnable now
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k':4})
    return retriever


# Retrieve relevant chunks from chroma
#retrieved_docs = retriever.invoke(question)


def join_the_retrieved_docs(retrieved_docs):
    joined_retrieved_docs = "\n".join(doc.page_content for doc in retrieved_docs)
    return joined_retrieved_docs

def context(inputs: dict):
    yt_link = inputs['yt_link']
    question = inputs['question']

    video_id = extract_youtube_id(yt_link)
    transcript = fetch_transcript(video_id)
    retriever = split_store_retrieve(transcript)
    retrieved_docs = retriever.invoke(question)
    return join_the_retrieved_docs(retrieved_docs)


# -----RETRIEVE AND AUGMENT-------


# Obtain history and question for passing to prompt
rag_query_inputs = RunnableParallel({
    "history": RunnableLambda(lambda x: get_memory_summary(x["yt_link"])),
    "question": RunnablePassthrough()
})

# Prompt to create standalone question from history and question
rag_query_rewrite_prompt = ChatPromptTemplate.from_template(
    """
    Reformulate the user question into a standalone query for the video transcript.
    Use conversation history for context.
    Conversation so far:
    {history}
    Question: {question}
    Return only the reformulated query.
    """
)

# Formulate a standalone question and retrieve context according to that
rag_context_chain = RunnableParallel({
                    "yt_link": RunnableLambda(lambda x: x["yt_link"]),
                    "question": (rag_query_inputs | rag_query_rewrite_prompt | llm | parser)
                    }) | RunnableLambda(context)




# PromptTemplate provides a structure to the prompt thats sent to the llm
RAG_prompt = PromptTemplate(
    template="""
      You are a strict assistant.
      Conversation so far:
      {history}
      Only use the transcript to answer the question. 
      - If the transcript does not explicitly contain the requested information, 
        you MUST reply with exactly: "I don't know."
      - Do not guess, infer, or add information that is not directly in the transcript.
      - Do not generalize. For example: If the transcript says "Turkey" but the question 
      asks "what part of Turkey", and no specific part is mentioned, 
      reply exactly with: "I don't know."
      Transcript: {context}
      Question: {question}
    """,
    input_variables=['history', 'context', 'question']
)


# parallel_chain prepares the structured output, produces {context, question}
parallel_chain = RunnableParallel({
    "history": RunnableLambda(lambda x: get_memory_summary(x["yt_link"])),
    "context": rag_context_chain,
    "question": RunnablePassthrough()
})


RAG_chain = parallel_chain | RAG_prompt | llm | parser




# ------------------------------------  WEB SEARCH FROM TAVILY   ---------------------------------------

# Mini prompt to expand query into a standalone web search query
query_rewrite_prompt = ChatPromptTemplate.from_template(
    """
    Reformulate the user question into a standalone web search query.
    Use conversation history for context if needed.
    Conversation so far:
    {history}
    Question: {question}
    Return only the reformulated query.
    """
)


search_tool = TavilySearch(
    max_results=5,
    topic='general',
    tavily_api_key = os.getenv('TAVILY_API_KEY')
)

tavily_prompt = PromptTemplate(
    template="""
    You are a helpful assistant.
    Use the following web search results to answer the question.
    If the information is not in the results, just say you don't know.
    Web Results: {context}
    Question: {question}
    """,
    input_variables=["context", "question"],
)


def format_results(response: dict) -> str:
  results = response.get('result',[])

  joined_content = ""
  for r in results:
    joined_content += r["content"] + "\n\n"
  return(joined_content)
    
standalone_query = RunnableParallel({
    'history': RunnableLambda(lambda x: get_memory_summary(x['yt_link'])),
    'question': RunnablePassthrough()
})

context_from_query = (standalone_query 
                        | query_rewrite_prompt 
                        | llm 
                        | parser 
                        | RunnableLambda(lambda q: {'query': q})
                        | search_tool 
                        | RunnableLambda(format_results))

tavily_parallel_chain = RunnableParallel({
    'context': context_from_query,
    'question': RunnablePassthrough()
})


tavily_chain = tavily_parallel_chain | tavily_prompt | llm | parser


# --------------------------------   ROUTE TO RAG OR WEB  ---------------------------------------------

def route(yt_link: str, question: str) -> str:
    video_id = extract_youtube_id(yt_link)
    memory = video_memories[video_id]
    
    rag_answer = RAG_chain.invoke({'yt_link':yt_link, 'question':question})
    if "i don't know" in rag_answer.lower():
        tavily_answer = tavily_chain.invoke({'yt_link': yt_link, 'question': question})
        final_answer = f"**Answering from the web**\n\n{tavily_answer}"
    else:
        final_answer = f"**Answering from the video**\n\n{rag_answer}"
    
    # UPDATE MEMORY with this turn
    memory.save_context({"input": question}, {"output": final_answer})
    return final_answer
