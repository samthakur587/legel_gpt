from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
app = FastAPI()
import uvicorn
import os
import openai
from langchain.llms import OpenAI
import PyPDF2
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
import shutil
#os.environ['OPENAI_API_KEY'] = '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'
openai_api_key = os.environ.get('OPENAI_API_KEY', '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')
llm = OpenAI(temperature=0.4)
from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo
)
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

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
path = ''
if not os.path.exists('uploads'):
    dir = os.mkdir('uploads')
@app.post("/legal_gpt")
async def chat_docs(question:str = 'explain about this case',file: UploadFile = File(...)):
    contents = await file.read()
    with open(f"uploads/{file.filename}", "wb") as f:
        f.write(contents) 
    path = f"uploads/{file.filename}"
    loader = PyPDFLoader(path)
    pages = loader.load_and_split()
    store = Chroma.from_documents(pages, embeddings, collection_name='legal_case_files')
    vectorstore_info = VectorStoreInfo(
    name="legal_case_files",
    description="Gpt on Legal Case Files",
    vectorstore=store)
    toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)
    agent_executor = create_vectorstore_agent(
            llm=llm,
            toolkit=toolkit,
            verbose=True
        )
    response = agent_executor.run(question)
    return {'message':response}
@app.get('/clearall')
def clear():
    shutil.rmtree('uploads')
    return {'message':'folder is deleted'}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
