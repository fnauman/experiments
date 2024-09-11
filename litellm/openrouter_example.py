from dotenv import load_dotenv
from litellm import completion


load_dotenv() # OPENROUTER_API_KEY is stored in .env file

response = completion(model="openrouter/google/gemma-7b-it:free", messages=[{
        "role": "user", 
        "content": "write code for saying hi"
    }]
)
print(response)
