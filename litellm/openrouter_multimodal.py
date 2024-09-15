import os
import base64
from openai import OpenAI
from dotenv import load_dotenv


load_dotenv() # OPENROUTER_API_KEY is stored in .env file

#  Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

# Path to your image
image_path = "/home/nauman/front_2023_03_23_14_49_40.jpg"

# Getting the base64 string
base64_image = encode_image(image_path)

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
        "text": "Describe this image?"
      },
      {
        "type": "image_url",
        # "image_url": {
        #     "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
        # }
        "image_url": {
            "url": f"data:image/jpeg;base64,{base64_image}"
        }

      }
    ]
  }
]
)
print(completion.choices[0].message.content)
