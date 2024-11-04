import json
from openai import OpenAI

client = OpenAI()

completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant answering questions .",
        },
        {"role": "user", "content": "Write a haiku about recursion in programming."},
    ],
)

print(completion)
