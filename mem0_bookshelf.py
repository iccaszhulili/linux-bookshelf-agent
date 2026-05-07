from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_ollama import OllamaEmbeddings

from typing import Annotated, TypedDict, List
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from mem0 import Memory
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()

model = init_chat_model(
    "deepseek-chat",
    model_provider="deepseek",
    temperature=0
)

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
            "url": "http://localhost:6333",
            "embedding_model_dims": 768,
        }
    }
}

mem0 = Memory.from_config(config)

# Create embedder
embeddings = OllamaEmbeddings(model='nomic-embed-text')

qdrant_client = QdrantClient(url="http://localhost:6333")
if not qdrant_client.collection_exists("bookshelf_docs"):
    from qdrant_client.models import Distance, VectorParams
    qdrant_client.create_collection(
        collection_name="bookshelf_docs",
        vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    )
vectorstore = QdrantVectorStore(
    client=qdrant_client,
    collection_name="bookshelf_docs",
    embedding=embeddings,
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

class State(TypedDict):
    messages: Annotated[List[HumanMessage | AIMessage], add_messages]
    mem0_user_id: str
    retrieved_docs: str
    memory_context: str

graph = StateGraph(State)

def chatbot(state: State):
    messages = state["messages"]
    user_id = state["mem0_user_id"]
    retrieved_docs = state["retrieved_docs"]
    memory_context = state["memory_context"]
 
    system_message = SystemMessage(content=f"""You are a linux knowledge assistant. Answer questions primarily from the document context. Use previous conversation context to understand follow-up questions and personalize responses. If the document doesn't cover the topic, say "I don't have the information in my docs".

Document context:
{retrieved_docs}

Previous conversation context:
{memory_context}
""")

    try:

        full_messages = [system_message] + messages
        response = model.invoke(full_messages)

        # Store the interaction in Mem0
        try:
            interaction = [
                {
                    "role": "user",
                    "content" : messages[-1].content
                },
                {
                    "role": "assistant",
                    "content": response.content
                }
            ]
            result = mem0.add(interaction, user_id=user_id)
            print(f"Memory saved: {len(result.get('results', []))} memories added")
        except Exception as e:
            print(f"Error saving memory: {e}")

        return {"messages": [response]}
    except Exception as e:
        print(f"Error in chatbot: {e}")
        # Fallback response without memory context
        response = model.invoke(messages)
        return {"messages": [response]}

def retrieve_docs(state: State):
    question = state["messages"][-1].content
    docs = retriever.invoke(question)

    formatted = ""
    for doc in docs:
        source = doc.metadata["source"]
        formatted += f"[Source: {source}]\n{doc.page_content}\n\n"

    return {"retrieved_docs": formatted}

def retrieve_memory(state: State):
    question = state["messages"][-1].content
    user_id = state["mem0_user_id"]

    # Retrieve relevant memories
    memories = mem0.search(question, filters={"user_id": user_id})
    # Handle dict response format
    memory_list = memories["results"]

    context = "Relevant information from previous conversations:\n"
    for memory in memory_list:
        context += f"- {memory['memory']}\n"
    print(f"context is {context}")

    return {"memory_context": context}

graph.add_node("chatbot", chatbot)
graph.add_node("retrieve_docs", retrieve_docs)
graph.add_node("retrieve_memory", retrieve_memory)
graph.add_edge(START, "retrieve_docs")
graph.add_edge("retrieve_docs", "retrieve_memory")
graph.add_edge("retrieve_memory", "chatbot")

compiled_graph = graph.compile(checkpointer=MemorySaver())

def run_conversation(user_input: str, mem0_user_id: str):
    config = {"configurable": {"thread_id": mem0_user_id}}
    state = {
        "messages": [HumanMessage(content=user_input)],
        "mem0_user_id": mem0_user_id
    } 

    for event in compiled_graph.stream(state, config):
        for value in event.values():
            if value.get("messages"):
                print("Customer Support:", value["messages"][-1].content)
                return

if __name__ == "__main__":
    compiled_graph.get_graph().draw_png("bookshelf_graph.png")
    print("Linux Assistant. Type 'quit' to exit.")
    user_id = "user1"
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("Customer Support: Thank you for reaching out to me. Have a good day!")
            break
        run_conversation(user_input, user_id)
