# Load secrets from file
from dotenv import load_dotenv
import os
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
conversation.run("")

# Chain
from langchain.chains import LLMChain, SimpleSequentialChain 

# Add a text input box for the user's question
user_question = "Malaga is an area with a lot of rain"

# Chain 1: Give the question to the AI, the output of the chain is the first response (statement)
template = """{question}\n\n"""
prompt_template = PromptTemplate(input_variables=["question"], template=template)
question_chain = LLMChain(llm=llm, prompt=prompt_template)

 # Chain 2: Generating assumptions made in the statement
template = """Here is a statement:
    {statement}
    Make a bullet point list of the assumptions you made when producing the above statement.\n\n"""
prompt_template = PromptTemplate(input_variables=["statement"], template=template)
assumptions_chain = LLMChain(llm=llm, prompt=prompt_template)
assumptions_chain_seq = SimpleSequentialChain(
    chains=[question_chain, assumptions_chain], verbose=True
)

# Chain 3: Fact checking the assumptions
template = """Here is a bullet point list of assertions:
{assertions}
For each assertion, determine whether it is true or false. If it is false, explain why.\n\n"""
prompt_template = PromptTemplate(input_variables=["assertions"], template=template)
fact_checker_chain = LLMChain(llm=llm, prompt=prompt_template)
fact_checker_chain_seq = SimpleSequentialChain(
    chains=[question_chain, assumptions_chain, fact_checker_chain], verbose=True
)

 # Final Chain: Generating the final answer to the user's question based on the facts and assumptions
template = """In light of the above facts, how would you answer the question '{}'. 
              Consider the facts and the assertion to give a good answer""".format(
    user_question
)
template = """{facts}\n""" + template
prompt_template = PromptTemplate(input_variables=["facts"], template=template)
answer_chain = LLMChain(llm=llm, prompt=prompt_template)
overall_chain = SimpleSequentialChain(
    chains=[question_chain, assumptions_chain, fact_checker_chain, answer_chain],
    verbose=True,
)

# Running all the chains on the user's question and displaying the final answer
print(overall_chain.run(user_question))