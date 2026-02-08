# llm.py
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

load_dotenv()

HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")


def load_llm(
    model_id= "openai/gpt-oss-120b",
    max_new_tokens=300,
    temperature=0.2,
):
    if not HUGGINGFACE_API_TOKEN:
        raise ValueError("HUGGINGFACEHUB_API_TOKEN not found in .env")

    endpoint = HuggingFaceEndpoint(
        repo_id=model_id,
        task="conversational",
        huggingfacehub_api_token=HUGGINGFACE_API_TOKEN,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )

    
    chat_llm = ChatHuggingFace(llm=endpoint)

    return chat_llm
