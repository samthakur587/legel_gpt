import os
from langchain.llms import OpenAI
import PyPDF2
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
os.environ['OPENAI_API_KEY'] = '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'
llm = OpenAI(temperature=0.4)
from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo
)
embeddings = OpenAIEmbeddings()
loader = PyPDFLoader('/home/samthakur/ubuntu_files/fastapi/2022_02_11_Montelogo_Ida.pdf')
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
prompt = input('ask a question .. ')
response = agent_executor.run(prompt)
