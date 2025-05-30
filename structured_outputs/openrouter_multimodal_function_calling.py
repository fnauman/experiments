import os
import base64
from openai import OpenAI
from dotenv import load_dotenv

from enum import Enum
from pydantic import BaseModel, Field
import instructor


load_dotenv("/home/nauman/.env") # OPENROUTER_API_KEY is stored in .env file

MODEL = "google/gemini-2.5-flash-preview-05-20"
# Path to your image
IMAGE_PATH = "/home/nauman/repos/private/clip_cisutac/sample/test1.jpg"

# Gets API Key from environment variable OPENAI_API_KEY
client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=os.getenv("OPENROUTER_API_KEY"),
)

#  Function to encode the image
def encode_image(IMAGE_PATH):
  with open(IMAGE_PATH, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')


# Getting the base64 string
base64_image = encode_image(IMAGE_PATH)


# Define the schema for the input data
class UsageEnum(str, Enum):
    """Defines the recommended use for the second hand clothing item at a sorting facility."""
    REUSE = "Reuse and Resell"
    REUSE_OUTSIDE = "Send clothes that are undesirable in Sweden to developing countries"
    OTHER = "Recycle, Repair or Dispose"

class DamageEnum(str, Enum):
    """Describes the nature and intensity of damage to the clothing item."""
    NONE = "No damage, like new condition"
    LOW = "Light damage, minor wear and tear"
    HIGH = "Heavy damage, significant wear and tear"

class TrendEnum(str, Enum):
    """Describes the trend of the clothing item."""
    BUSINESS = "Business"
    CLASSIC = "Classic"
    VINTAGE = "Vintage"
    SPORTY = "Sporty"
    CASUAL = "Casual"
    OTHER = "Other"

class ClothingItem(BaseModel):
    """Characteristics of a second hand clothing item: Pattern, Style, Category, Type, and Usage."""
    usage: UsageEnum = Field(..., description="Recommended use for the clothing item") # Field(None, description="") was leading to usage=None 
    caption: str = Field(..., description="Description of the clothing item in a sentence")
    confidence_score: float = Field(..., description="The confidence score of the prediction of usage as a percentage")
    pattern: str = Field(..., description="Clothing patterns such as stripes, floral, or solid")
    trend: TrendEnum = Field(..., description="Trend of the clothing item")
    damage: DamageEnum = Field(..., description="Characterizes the nature and intensity of damage to the clothing item")
    type: str = Field(..., description="Clothing category such as shirt, pants, or dress")
    category: str = Field(..., description="Clothing type such as men, women, or children")


def annotate_front_img(base64_img: str) -> ClothingItem:
    # response = instructor.from_openai(OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))).chat.completions.create(
    response = instructor.from_openai(client, mode=instructor.Mode.JSON).chat.completions.create(
        model=MODEL, # 'gpt-4-turbo',
        response_model=ClothingItem, # EmployeeList,
        # seed=SEED, 
        # temperature=TEMPERATURE, 
        # max_tokens=300, 
        messages=[
            {
                "role": "user",
                "content": 'Analyze the image of a clothing item to identify its attributes and recommend the optimal usage.',
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_img}", 
                            "detail": "low"
                        }
                    },
                ],
            }
        ],
    )
    return response

response = annotate_front_img(base64_image)
print(response.model_dump_json(indent=4))
