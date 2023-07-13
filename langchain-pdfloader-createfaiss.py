
'''
https://techcommunity.microsoft.com/t5/startups-at-microsoft/build-a-chatbot-to-query-your-documentation-using-langchain-and/ba-p/3833134
Done only once:
Load files > Split text > Create embeddings > Save to FAISS
'''
import os
from dotenv import load_dotenv
from langchain import OpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# Set openAI key and create llm
load_dotenv('secrets.env')
OPENAI_API_KEY = os.getenv('OPENAI_KEY')
llm = OpenAI(openai_api_key=OPENAI_API_KEY, temperature=0.9)

# 1) Load files > Split text > Create embeddings > Save to FAISS
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, chunk_size=1)
dataPath = "./data/profiles/"
fileName = dataPath + "perfiljuan.pdf"

#use langchain PDF loader
loader = PyPDFLoader(fileName)

#split the document into chunks
pages = loader.load_and_split()

#Use Langchain to create the embeddings using text-embedding-ada-002
db = FAISS.from_documents(documents=pages, embedding=embeddings)

#save the embeddings into FAISS vector store
db.save_local("./data/faiss_index")