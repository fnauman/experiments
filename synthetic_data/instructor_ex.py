from typing import Iterable
from pydantic import BaseModel
import instructor
from openai import OpenAI


# Define the UserDetail model
class UserDetail(BaseModel):
    name: str
    age: int


# Patch the OpenAI client to enable the response_model functionality
client = instructor.from_openai(OpenAI())


def generate_fake_users(count: int) -> Iterable[UserDetail]:
    return client.chat.completions.create(
        model="gpt-3.5-turbo",
        response_model=Iterable[UserDetail],
        messages=[
            {"role": "user", "content": f"Generate a {count} synthetic users"},
        ],
    )


for user in generate_fake_users(5):
    print(user)
    """
    name='Alice' age=25
    name='Bob' age=30
    name='Charlie' age=35
    name='David' age=40
    name='Eve' age=45
    """
