from fastapi import FastAPI
app = FastAPI()
import uvicorn
import os

from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
import openai
>>>>>>> 68e737ab2c3d13e0c1dfb025935e32284836ff97
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

@app.get("/")
def read_root():
    return HTMLResponse(
        """
        <html>
            <body>
                <h1>Upload a PDF file</h1>
                <form action="/upload" enctype="multipart/form-data" method="post">
                    <input type="file" name="file" accept=".pdf">
                    <input type="submit">
                </form>
            </body>
        </html>
        """
    )
if not os.path.exists('uploads'):
    dir = os.mkdir('uploads')
@app.get("/")
def read_root():
    return {'message':'welcome to legel chat gpt'}
@app.post("/upload_text")
async def input_text(text:str):
    with open('uploads/text.txt','w') as f:
        f.write(text)
    loader = TextLoader('uploads/text.txt')
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
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
    agent = agent_executor
    return {'message':'uploaded seccessfully'}
def run():
    loader = TextLoader('uploads/text.txt')
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
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
    return agent_executor
@app.post("/ask_qa")
async def ask(question:str='explain this case'):
    if len(os.listdir('uploads')) !=0:

        agent = run()
        response = agent.run(question)
        return {'answer':response}
    else: return {'message':'upload the text first'}
    
@app.get('/clearuploads')
def clear():
    shutil.rmtree('uploads')
    return {'message':'cleaned all the memory'}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
