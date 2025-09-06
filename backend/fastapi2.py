from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse


from pipeline import RAG_chain, tavily_chain, route

import os
import re

# ---------------------------------   LANGSMITH TRACING   -------------------------------------------------
os.environ['LANGCHAIN_TRACING_V2'] = os.getenv('LANGCHAIN_TRACING_V2')
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_PROJECT'] = os.getenv('LANGCHAIN_PROJECT')


# -------------------------------------------  CREATING THE APP   --------------------------------------

app = FastAPI()

#-------------------------------  VALIDATOIN  -----------------------------------------
class QuestionRequest(BaseModel):
    url: str
    question: str

class AnswerResponse(BaseModel):
    answer: str

# -------------------------------------  API  ---------------------------------------------

@app.post('/chat', response_model=AnswerResponse)
def answer(req: QuestionRequest):
    try:
        answer = route(req.url, req.question )
        return {'answer': answer}
    except ValueError as e:
        return JSONResponse(status_code=422, content={"answer": f"{str(e)}"})

if __name__ == '__main__':
    import uvicorn
    uvicorn.run("fastapi2:app", host="0.0.0.0", port=1111, reload=True)