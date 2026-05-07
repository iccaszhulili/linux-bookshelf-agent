import gradio as gr
from langchain_core.messages import HumanMessage
from mem0_bookshelf import compiled_graph

user_id = "user1"
last_memory = ""

def respond(user_input, history):
    global last_memory
    config = {"configurable":{"thread_id": user_id}}
    
    if user_input.strip() == "/memory":
        if last_memory:
            return f"<span style='color: gray'>{last_memory}</span>"    
        return "No memory context available."

    state = {
        "messages": [HumanMessage(content=user_input)],
        "mem0_user_id": user_id
    }

    response_text = ""
    for event in compiled_graph.stream(state, config):
        for value in event.values():
            if value.get("messages"):
                response_text = value["messages"][-1].content
            if value.get("memory_context"):
                last_memory = value["memory_context"]

    if not response_text:
        return "No response generated."
    return f"**{response_text}**"

# Create and launch Gradio interface
demo = gr.ChatInterface(fn=respond, title="Linux Knowledge Assistant")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=False)
