import os
from openai import OpenAI
from dotenv import load_dotenv


load_dotenv() # OPENROUTER_API_KEY is stored in .env file

# gets API Key from environment variable OPENAI_API_KEY
client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=os.getenv("OPENROUTER_API_KEY"),
)

completion = client.chat.completions.create(
#   extra_headers={
#     "HTTP-Referer": $YOUR_SITE_URL, # Optional, for including your app on openrouter.ai rankings.
#     "X-Title": $YOUR_APP_NAME, # Optional. Shows in rankings on openrouter.ai.
#   },
  model="mistralai/pixtral-12b:free",
  messages=[
  {
    "role": "user",
    "content": [
      {
        "type": "text",
        "text": "What's in this image?"
      },
      {
        "type": "image_url",
        "image_url": {
          "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
        }
      }
    ]
  }
]
)
print(completion.choices[0].message.content)
