import os
# import autogen
from dotenv import load_dotenv
from autogen import ConversableAgent


load_dotenv() # OPENROUTER_API_KEY is stored in .env file

llm_config = {
    "config_list": [{
        "model": "google/gemma-7b-it:free",
        "api_key": os.environ["OPENROUTER_API_KEY"],
        "base_url": "https://openrouter.ai/api/v1"
    }]
}

agent = ConversableAgent(
    "chatbot",
    llm_config=llm_config,
    code_execution_config=False,  # Turn off code execution, by default it is off.
    function_map=None,  # No registered functions, by default it is None.
    human_input_mode="NEVER",  # Never ask for human input.
)
reply = agent.generate_reply(messages=[{
        "role": "user", 
        "content": "Tell me a joke."
    }]
)
print(reply)
