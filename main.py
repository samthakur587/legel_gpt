from fastapi import FastAPI
app = FastAPI()
import uvicorn
from pydantic import BaseModel
import os

from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
import openai
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
import shutil

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm = OpenAI(temperature=0.4)
from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo
)

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY )

class TextData(BaseModel):
    id: str
    text: str

if not os.path.exists('uploads'):
    os.mkdir('uploads')


@app.get("/")
def read_root():
    return {'message':'welcome to legel gpt'}


@app.post("/upload_text")
async def input_text(data: TextData):
    try:
        with open(f'uploads/text.txt', 'w') as f:
            f.write(data.text)
        
        loader = TextLoader('uploads/text.txt')
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=700, chunk_overlap=0)
        docs = text_splitter.split_documents(documents)
        store = Chroma.from_documents(docs, embeddings, collection_name='legal_case_files')
        vectorstore_info = VectorStoreInfo(
        name="legal_case_files",
        description="Gpt on Legal Case Files",
        vectorstore=store)
        toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)
    
        agent_executor = create_vectorstore_agent(
            llm=llm,
            toolkit=toolkit,
            verbose=True)
        app.state.agent = agent_executor
        return {'message':'uploaded seccessfully'}
    except Exception as e:
        return {'message': 'Upload failed', 'error': str(e)}


# def run():
#     loader = TextLoader('uploads/text.txt')
#     documents = loader.load()
#     text_splitter = CharacterTextSplitter(chunk_size=700, chunk_overlap=0)
#     docs = text_splitter.split_documents(documents)
#     store = Chroma.from_documents(docs, embeddings, collection_name='legal_case_files')
#     vectorstore_info = VectorStoreInfo(
#     name="legal_case_files",
#     description="Gpt on Legal Case Files",
#     vectorstore=store)
#     toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)

#     agent_executor = create_vectorstore_agent(
#             llm=llm,
#             toolkit=toolkit,
#             verbose=True)
#     return agent_executor
@app.post("/ask_qa")
async def ask( data: TextData):
    try:
        if len(os.listdir('uploads')) !=0:
            agent = app.state.agent
            # agent = run()
            response = agent.run(data.text)
            return {'message': 'Question answered', 'id': data.id, 'answer': response}
        else: 
            return {'message':'upload the text first'}
    except Exception as e:
        return {'message': 'Question answering failed', 'id': data.id, 'error': str(e)}
    
@app.get('/clear_uploads')
def clear_uploads():
    try:
        shutil.rmtree('uploads')
        os.mkdir('uploads')
        return {'message': 'Uploads cleared successfully'}
    except Exception as e:
        return {'message': 'Uploads clearing failed', 'error': str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
