# My Open AI and LangChain playground 🛴

## Chains example

The user will give a question or statement to the AI. The AI will not give back the first answer. It will create a list of assumptions, assert them, and then give a better response.

```python
user_question = "Malaga is an area with a lot of rain"
```

## Chain 1: Give the question to the AI, the output of the chain is the first response (statement)
```python
template = """{question}\n\n"""
prompt_template = PromptTemplate(input_variables=["question"], template=template)
question_chain = LLMChain(llm=llm, prompt=prompt_template)
```
**chain output, first AI response:**

No, Malaga is a city in Spain located on the south coast of the Mediterranean Sea. It has a Mediterranean climate with hot dry summers and mild, wet winters. The average annual temperature is 19.1 °C (66.3°F). Rainfall is low, with an average annual precipitation of only 331 mm (13.03 inches).

## Chain 2: Generating assumptions made in the statement
```python
template = """Here is a statement:
    {statement}
    Make a bullet point list of the assumptions you made when producing the above statement.\n\n"""
prompt_template = PromptTemplate(input_variables=["statement"], template=template)
assumptions_chain = LLMChain(llm=llm, prompt=prompt_template)
assumptions_chain_seq = SimpleSequentialChain(
    chains=[question_chain, assumptions_chain], verbose=True
)
```

**chain output, assumptions:**

- Malaga is a city in Spain
- Located on the south coast of the Mediterranean Sea  
- Mediterranean climate
- Hot dry summers
- Mild, wet winters
- Average annual temperature of 19.1 °C (66.3°F)       
- Average annual precipitation of 331 mm (13.03 inches)

## Chain 3: Fact checking the assumptions
```python
template = """Here is a bullet point list of assertions:
{assertions}
For each assertion, determine whether it is true or false. If it is false, explain why.\n\n"""
prompt_template = PromptTemplate(input_variables=["assertions"], template=template)
fact_checker_chain = LLMChain(llm=llm, prompt=prompt_template)
fact_checker_chain_seq = SimpleSequentialChain(
    chains=[question_chain, assumptions_chain, fact_checker_chain], verbose=True
)
```

**chain output, assert assumptions:**

- Malaga is a city in Spain: True. Malaga is a city located in the autonomous community of Andalucía in the country of Spain.
- Located on the south coast of the Mediterranean Sea: True. Malaga is located on the south coast of the Mediterranean Sea.
- Mediterranean climate: True. Malaga has a Mediterranean climate, characterized by hot, dry summers and mild wet winters.
- Hot dry summers: True. Malaga's summers are usually dry and hot, with temperatures reaching up to 33 °C (91 °F).
- Mild, wet winters: True. Malaga's winters are usually mild, with temperatures rarely dropping below 10 °C (50 °F) and wet, with an average precipitation of 331 mm (13.03 inches) per year.
- Average annual temperature of 19.1 °C (66.3°F): False. Malaga's average annual temperature is 17.4 °C (63.3°F).
- Average annual precipitation of 331 mm (13.03 inches): True

## Final Chain: Generating the final answer to the user's question based on the facts and assumptions
```python
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
```

**final answer:**

No, Malaga does not have a lot of rain, with an average annual precipitation of 331 mm (13.03 inches). However, the area does experience mild, wet winters which can bring lots of rainfall in the short-term.
