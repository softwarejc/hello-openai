import os
from dotenv import load_dotenv
from langchain import OpenAI, ConversationChain, LLMChain, PromptTemplate

# Set openAI key and create llm
load_dotenv('secrets.env')
llm = OpenAI(openai_api_key=os.getenv('OPENAI_KEY'), temperature=0.9)

# Template
template = "What are the top 3 books about {topic}"
prompt = PromptTemplate(template=template,input_variables=['topic'])
chain = LLMChain(llm=llm,prompt=prompt)
input = {'topic':'personal development'}
print(chain.run(input))

# Conversation
conversation = ConversationChain(llm=llm, verbose=True)
conversation.run("Hi there, my name is Juan Carlos!")
conversation.run("Do you remember my name?")
conversation.run("bye")