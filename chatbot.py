import gradio as gr
from langchain_core.messages import HumanMessage
from agentic_rag import graph
from mem0_config import search_memory

user_id = "user1"

def respond(user_input, history):
    config = {"configurable":{"thread_id": user_id}}
    
    if user_input.strip() == "/memory":
        try:
            memory = search_memory("conversation history", user_id)
            if memory and memory != "No relevant memories found.":
                return f"<span style='color: gray'>{memory}</span>" 
        except Exception as e:
            return f"Memory search failed: {e}"   
        return "No relevant memories found."

    state = {
        "messages": [HumanMessage(content=user_input)],
    }

    response_text = ""
    for event in graph.stream(state, config):
        for value in event.values():
            if value.get("messages"):
                response_text = value["messages"][-1].content

    if not response_text:
        return "No response generated."
    return f"**{response_text}**"

# Create and launch Gradio interface
demo = gr.ChatInterface(fn=respond, title="Linux Knowledge Assistant")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=False)
