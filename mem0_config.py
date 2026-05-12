import os
from mem0 import Memory
from dotenv import load_dotenv

load_dotenv()

config = {
    "llm": {
        "provider": "litellm",
        "config": {
            "model": "deepseek/deepseek-chat",
            "temperature": 0,
        }
    },
    "embedder":{
        "provider": "ollama",
        "config": {
            "model": "nomic-embed-text",
        }
    },
    "vector_store":{
        "provider": "qdrant",
        "config": {
            "collection_name": "mem0",
            "url": os.getenv("QDRANT_URL", "http://localhost:6333"),
            "embedding_model_dims": 768,
        }
    }
}

mem0 = Memory.from_config(config)

def search_memory(question, user_id):
    memories = mem0.search(question, filters={"user_id": user_id})
    memory_list = memories["results"]
    context = ""
    for memory in memory_list:
        context += f"- {memory['memory']}\n"
    return context if context else "No relevant memories found."

def save_memory(messages, user_id):
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    elif not isinstance(messages, list):
        raise ValueError(f"messages must be a str or a list, but got {str(messages)}")
    mem0.add(messages, user_id=user_id)