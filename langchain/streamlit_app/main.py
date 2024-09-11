from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv

load_dotenv()

# openai = OpenAI()


""" Introducing a PromptTemplate"""
def v2_generate_computer_names(field="astrophysics"):

    llm = OpenAI(temperature=0.6)

    prompt_template_field = PromptTemplate(
        input_variables=["field"], 
        template="""
            I bought a new computer. I want to create an interesting name for it. 
            Please suggest 5 different names for my computer. 
            The names should be inspired by my background in {field}.""", 
    )

    # LLMChain(llm=llm, prompt=prompt_template_field)
    name_chain = LLMChain(llm=llm, prompt=prompt_template_field, output_key="computer_name") 

    response = name_chain({"field": field})

    return response["computer_name"] # response['text'] # response returns a JSON object




""" Only using the LLM wrapper from Langchain
"""
def v1_generate_computer_names():
    llm = OpenAI(temperature=0.6)

    name = llm("I bought a new computer. I want to create an interesting name for it. Please suggest 5 different names for my computer. The names should be inspired by my background in astrophysics.")

    return name


if __name__ == "__main__":
    # print(v1_generate_computer_names())
    print(v2_generate_computer_names(field="computer science"))
