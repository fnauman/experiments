from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from dotenv import load_dotenv

import requests
import bs4

load_dotenv()


template = """Summarize the following question based on the context: 

Question: {question}

Context: 

{context}"""

prompt = ChatPromptTemplate.from_template(template)

url = "https://blog.langchain.dev/announcing-langsmith/"


def scrape_text(url: str):
    res = requests.get(url)
    soup = bs4.BeautifulSoup(res.text, "html.parser")
    # text = soup.find("div", {"class": "post-content"}).text
    text = soup.get_text(separator=" ", strip=True)
    return text

page_content = scrape_text(url)[:10000]

model = "gpt-3.5-turbo-1106"

chain = prompt | ChatOpenAI(model=model) | StrOutputParser()

chain.invoke(
    {
        "question": "What is langsmith?", 
        "context": page_content
    }
)

print(chain.)