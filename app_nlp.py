import os
from langchain.llms import OpenAI
import PyPDF2
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
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
text = "page_content='WORKERS’ COMPENSATION APPEALS BOARD \nSTATE OF CALIFORNIA \nIDA MONTELONGO , Applicant  \nvs. \nGELSONS MARKET ;  \nEVEREST NATIONAL INSURANCE COMPANY, administered by  \nAMERICAN CLAIMS MANAGEMENT , Defendants  \nAdjudication Number:  ADJ2193346 \nMarina Del Rey  District Office  \n \nOPINION AND ORDER \nDISMISSING PETITION FOR RECONSIDERATION \n Lien claimant Charles Schwarz, M.D., seeks reconsideration of the Findings of Fact \nRegarding Liens issued on September 1, 2021 by a workers’ compensation administrative law \njudge (WCJ). The WCJ found that lien claimant was required to file a declaration pursuant to \nLabor Code1 section 4903.05, but failed to do so, and thus, that lien claimant’s lien is invalid.   \n Lien claimant contends that he was not required to file a declaratio n pursuant to section \n4903.05 because the lien was filed in 2003 and was thus, never subject to a filing fee under section 4903.05, subdivision (c)(2). \n Defendant filed an Answer to Petition for Reconsideration (Answer), and the WCJ filed a \nReport and Reco mmendation on Petition for Reconsideration (Report). The WCJ recommends that \nthe Appeals Board denies the Petition for Reconsideration.  \n We have reviewed the record in this case, the allegations of the Petition for Reconsideration \nand the Answer, as well as the contents of the Report. Based on the reasons set forth below, we \ndismiss the Petition for Reconsideration.  \n                                                 \n1 All further references are to the Labor Code unless otherwise noted.' metadata={'source': '/home/samthakur/ubuntu_files/fastapi/2022_02_11_Montelogo_Ida.pdf', 'page': 0}"
with open('text.txt','w') as f:
    f.write(text)

embeddings = OpenAIEmbeddings()
loader = TextLoader('text.txt')
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
            verbose=True
        )
print(type(vectorstore_info))
#prompt = input('ask a question .. ')
#response = agent_executor.run(prompt)
