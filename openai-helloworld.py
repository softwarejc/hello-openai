# OpenAI libraries
import openai 
# Load secrets from file
from dotenv import load_dotenv
import os

load_dotenv('secrets.env')
openai.api_key = os.getenv('OPENAI_KEY')

response = openai.Completion.create(
  engine="text-davinci-003",
  prompt="Explain machine learning in one paragraph",
  temperature=0.5,
  max_tokens=100
)

print(response.choices[0].text.strip())
