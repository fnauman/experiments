# GPT-4o-2024-08-06: $0.000638 per image
# GPT-4.1: $0.00051 per image
# o4-mini: $0.000485 per image
import os
import base64
from PIL import Image
from io import BytesIO
from openai import OpenAI
from dotenv import load_dotenv
from enum import Enum
from pydantic import BaseModel, Field
from typing import List
import json

# Load environment variables
load_dotenv("../.env")

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Note: gpt-4.1 has limited structured output support, using gpt-4o-2024-08-06 instead
MODEL = "gpt-4o-2024-08-06"

# Define the allowed values as Enums for strict validation
class Color(str, Enum):
    BLACK = "black"
    WHITE = "white" 
    RED = "red"
    GREEN = "green"
    BLUE = "blue"
    YELLOW = "yellow"
    BROWN = "brown"
    ORANGE = "orange"
    PURPLE = "purple"
    PINK = "pink"
    GRAY = "gray"
    BEIGE = "beige"
    METALLIC = "metallic"
    MULTICOLOR = "multicolor"

class Trend(str, Enum):
    ATHLETIC_SPORTY = "athletic"
    CASUAL = "casual"
    FORMAL = "formal"
    STREETWEAR = "streetwear"
    VINTAGE = "vintage"
    CLASSIC = "classic"
    ETHNIC_TRADITIONAL = "traditional"

class Category(str, Enum):
    MENS = "men's"
    WOMENS = "women's"
    KIDS = "kid's"
    UNISEX = "unisex"

class Price(str, Enum):
    BUDGET = "budget"
    MID_RANGE = "mid-range"
    PREMIUM = "premium"

class GarmentAnalysis(BaseModel):
    """Analysis of a garment image with four key attributes"""
    color: Color = Field(..., description="Primary color of the garment")
    trend: Trend = Field(..., description="Style trend category of the garment")
    category: Category = Field(..., description="Target demographic category")
    price: Price = Field(..., description="Price range classification")

def resize_image_to_512x512(image_path: str) -> str:
    """
    Resize image to 512x512 pixels and return base64 encoded string.
    This helps reduce API costs as specified in the requirements.
    """
    # Open and resize image
    with Image.open(image_path) as img:
        # Convert to RGB if necessary (handles RGBA, grayscale, etc.)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize to 512x512 using high-quality resampling
        img_resized = img.resize((512, 512), Image.Resampling.LANCZOS)
        
        # Save to BytesIO buffer
        buffer = BytesIO()
        img_resized.save(buffer, format='JPEG', quality=95)
        buffer.seek(0)
        
        # Convert to base64
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
    return img_base64

def analyze_garment_image(image_path: str) -> GarmentAnalysis:
    """
    Analyze a garment image using OpenAI's structured outputs with strict mode.
    
    Args:
        image_path: Path to the garment image file
        
    Returns:
        GarmentAnalysis: Structured analysis with color, trend, category, and price
    """
    
    # Resize image to 512x512 to reduce costs
    base64_image = resize_image_to_512x512(image_path)
    
    # Use the newer chat completions API with structured outputs
    response = client.beta.chat.completions.parse(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": """You are an expert fashion analyst. Analyze the garment in the image and classify it according to the four attributes provided. 

For color: Choose the primary/dominant color visible in the garment.
For trend: Determine the style category based on the garment's design, cut, and overall aesthetic.
For category: Identify the target demographic this garment is designed for.
For price: Estimate the price range based on visible quality indicators like fabric, construction, brand markers, and overall design sophistication."""
            },
            {
                "role": "user", 
                "content": [
                    {
                        "type": "text",
                        "text": "Please analyze this garment image and provide the color, trend, category, and price classification."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "auto"  # Use auto detail for balanced cost/quality
                        }
                    }
                ]
            }
        ],
        response_format=GarmentAnalysis,
        temperature=0.1,  # Low temperature for consistent classification
        max_tokens=1000
    )
    
    return response.choices[0].message.parsed

def analyze_garment_image_with_json_schema(image_path: str) -> dict:
    """
    Alternative implementation using JSON schema with strict=True instead of Pydantic parsing.
    This demonstrates the direct JSON schema approach with strict structured outputs.
    """
    
    base64_image = resize_image_to_512x512(image_path)
    
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": """You are an expert fashion analyst. Analyze the garment in the image and classify it according to the four attributes provided. 

For color: Choose the primary/dominant color visible in the garment.
For trend: Determine the style category based on the garment's design, cut, and overall aesthetic.
For category: Identify the target demographic this garment is designed for.
For price: Estimate the price range based on visible quality indicators like fabric, construction, brand markers, and overall design sophistication."""
            },
            {
                "role": "user", 
                "content": [
                    {
                        "type": "text",
                        "text": "Please analyze this garment image and provide the color, trend, category, and price classification."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "auto"
                        }
                    }
                ]
            }
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "garment_analysis",
                "strict": True,  # Enable strict mode for guaranteed compliance
                "schema": {
                    "type": "object",
                    "properties": {
                        "color": {
                            "type": "string",
                            "enum": [
                                "black", "white", "red", "green", "blue", "yellow", 
                                "brown", "orange", "purple", "pink", "gray", "beige", 
                                "metallic", "multicolor"
                            ]
                        },
                        "trend": {
                            "type": "string", 
                            "enum": [
                                "athletic", "casual", "formal", "streetwear",
                                "vintage", "classic", "traditional"
                            ]
                        },
                        "category": {
                            "type": "string",
                            "enum": ["men's", "women's", "kid's", "unisex"]
                        },
                        "price": {
                            "type": "string",
                            "enum": ["budget", "mid-range", "premium"]
                        }
                    },
                    "required": ["color", "trend", "category", "price"],
                    "additionalProperties": False
                }
            }
        },
        temperature=0.1,
        max_tokens=1000
    )
    
    import json
    return json.loads(response.choices[0].message.content)

# Example usage
if __name__ == "__main__":
    # Replace with your image path
    image_path = "../clip/sample/test1.jpg"
    
    if os.path.exists(image_path):
        try:
            print("=" * 50)
            print("GARMENT ANALYSIS WITH STRUCTURED OUTPUTS")
            print("=" * 50)
            
            result = analyze_garment_image(image_path)
            
            print("\n=== GARMENT ANALYSIS RESULTS ===")
            print(f"Color: {result.color}")
            print(f"Trend: {result.trend}")
            print(f"Category: {result.category}")
            print(f"Price: {result.price}")
            
            print("\n=== JSON OUTPUT ===")
            print(result.model_dump_json(indent=2))
            
            print("\n=== ALTERNATIVE JSON SCHEMA APPROACH ===")
            json_result = analyze_garment_image_with_json_schema(image_path)
            print(json.dumps(json_result, indent=2))
            
            print("\n" + "=" * 50)
            print("💡 For cost analysis, run: python cost_calculator.py")
            print("=" * 50)
            
        except Exception as e:
            print(f"Error analyzing image: {e}")
            print("Make sure you have set OPENAI_API_KEY in your .env file")
    else:
        print(f"\nImage file not found: {image_path}")
        print("Please update the image_path variable with a valid image file.")
        print("\n💡 For cost analysis, run: python cost_calculator.py") 
    