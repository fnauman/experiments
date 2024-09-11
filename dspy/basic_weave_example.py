import os
import dspy
import weave
from dotenv import load_dotenv


load_dotenv() # OPENROUTER_API_KEY is stored in .env file


weave.init(project_name="dspy-intro")

gpt3_turbo = dspy.OpenAI(model="gpt-3.5-turbo-1106", max_tokens=300)
dspy.configure(lm=gpt3_turbo)
classify = dspy.Predict("sentence -> sentiment")
classify(sentence="it's a charming and often affecting journey.")
