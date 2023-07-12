# Langchain
from langchain.llms import OpenAI
# Load secrets from file
from dotenv import load_dotenv
import os

# Set openAI key
load_dotenv('secrets.env')
key = os.getenv('OPENAI_KEY')

#MORE temperature = more random
llm = OpenAI(openai_api_key=key, temperature=0.9)

prediction = llm.predict("What would be a good company name for a company that makes colorful socks?")
print(prediction)

