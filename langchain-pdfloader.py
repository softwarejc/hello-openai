'''
# https://techcommunity.microsoft.com/t5/startups-at-microsoft/build-a-chatbot-to-query-your-documentation-using-langchain-and/ba-p/3833134
# When a user asks a question, we will use the FAISS vector index to find the closest matching text.
# Feed that into GPT-3.5 as context in the prompt
# GPT-3.5 will generate an answer that accurately answers the question.

Steps:
FAISS index is loaded into RAM
User asks a question
User's question is sent to the OpenAI Embeddings API, which returns a 1536 dimensional vector.
The FAISS index is queried for the closest matching vector.
The closest matching vector is returned, along with the text that it was generated from.
The returned text is fed into GPT-35 as context in a GPT-35 prompt
GPT-35 generates a response, which is returned to the user.
'''

import os
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
import openai
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.chat_models import AzureChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings

def ask_question(qa, question):
    result = qa({"query": question})
    print("Answer:", result["result"])

# Set openAI key and create llm
load_dotenv('secrets.env')
OPENAI_API_KEY = os.getenv('OPENAI_KEY')
llm = OpenAI(openai_api_key=OPENAI_API_KEY, temperature=0.9)
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, chunk_size=1)

#load the faiss vector store we saved into memory
vectorStore = FAISS.load_local("./data/faiss_index", embeddings)

#use the faiss vector store we saved to search the local document
retriever = vectorStore.as_retriever(search_type="similarity", search_kwargs={"k":2})

#use the vector store as a retriever
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=False)
while True:
    query = input('you: ')
    if query == 'q':
        break
    ask_question(qa, query)