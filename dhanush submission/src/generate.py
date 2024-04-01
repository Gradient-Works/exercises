import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

def generate_response(query, extracted_docs):
    """
    Generates a response using the OpenAI GPT-3.5 Turbo model.

    Args:
        query (str): The user's query.
        extracted_docs (str): The extracted documents related to the query.

    Returns:
        str: The generated response from the GPT-3.5 Turbo model.
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    llm_content = query + '\n\n' + extracted_docs
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        temperature=0.8,
        messages=[
            {"role": "system", "content": "You are an helpful assistant. You are helping a user with a question. \
            If you dont know the answer just say I dont know instead of making stuff up and use only\
            the given data to answer the query. If the user asks generic question \
            which is not related to the content, politely ask them to stay within the context."},
            {"role": "user", "content": llm_content}
        ]
    )
    return (response.choices[0].message.content)
