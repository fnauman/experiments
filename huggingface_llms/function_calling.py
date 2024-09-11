# Instructor does not seem to support hugginface transformer models hosted locally
# Outlines, in contrast, does!
# pip install outlines

import torch
import outlines
from transformers import AutoModelForCausalLM, AutoTokenizer #, pipeline


# "microsoft/Phi-3.5-vision-instruct"
model_name = "microsoft/Phi-3.5-vision-instruct" # "gpt2"

torch.random.manual_seed(0) 
model_hf = AutoModelForCausalLM.from_pretrained( 
    model_name,  
    device_map="cuda",  
    torch_dtype="auto",  
    trust_remote_code=True,  
) 

tokenizer = AutoTokenizer.from_pretrained(model_name)

# model = outlines.models.transformers(model_name)

# NOTE Capital "T" in Transformers
model = outlines.models.Transformers(model_hf, tokenizer) 

prompt = """You are a sentiment-labelling assistant.
Is the following review positive or negative?

Review: This restaurant is just awesome!
"""

generator = outlines.generate.choice(model, ["Positive", "Negative"])
answer = generator(prompt)
print(answer)