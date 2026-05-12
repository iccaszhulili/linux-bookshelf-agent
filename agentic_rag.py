import os
from pydantic import BaseModel, Field
from typing import Literal
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain.tools import tool
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain.chat_models import init_chat_model
from qdrant_client import QdrantClient
from dotenv import load_dotenv
from mem0_config import search_memory, save_memory

load_dotenv()


def get_last_question(messages):
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return m.content
    return messages[0].content

model = init_chat_model(
    "deepseek-chat",
    model_provider="deepseek",
    temperature=0
)

embeddings = OllamaEmbeddings(model='nomic-embed-text')

client = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))
vectorstore = QdrantVectorStore(
    client=client,
    collection_name="bookshelf_docs",
    embedding=embeddings,
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

@tool
def retrieve_docs(query: str) -> str:
    """Search and return information about Linux topics."""
    docs = retriever.invoke(query)
    return "\n\n".join([doc.page_content for doc in docs])

def generate_query_or_respond(state: MessagesState):
    """Call the model to generate a response based on the current state."""
    messages = state["messages"]
    user_id = "user1"
    # If last message is from rewrite_question (AIMessage without tool calls),
    # force retrieval with the rewritten question.

    if len(messages) > 1 and isinstance(messages[-1], AIMessage) and not messages[-1].tool_calls:
        rewritten_msg = messages[-1].content
        return {"messages": [AIMessage(
            content="",
            tool_calls=[{
                "id": "forced_retrieval",
                "name": "retrieve_docs",
                "args": {"query": rewritten_msg}
                }]
            )]
        }

    response = model.bind_tools([retrieve_docs]).invoke(state["messages"])

    # If LLM responds directly (no Rtool call), save to memory
    if not response.tool_calls:
        save_memory(
            [
                {"role": "user", "content": messages[-1].content},
                {"role": "assistant", "content": response.content}
            ],
            user_id
        )
    return {"messages": [response]}

GRADE_PROMPT = (
    "You are a grader assessing relevance of a retrieved document to a user question. \n"
    "Here is the retrieved document: \n\n {context} \n\n"
    "Here is the user question: {question}\n"
    "If the document contains the keywords or semantic meaning related to the user question, grade it as relevant. \n"
    "Give the binary score 'yes' or 'no' to indicate whether the document is relevant to the question."
)

class GradeDocument(BaseModel):
    """Grade documents using a binary score for relevance check."""
    binary_score: str = Field(
        description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
    )

def grade_documents(state: MessagesState) -> Literal["generate_answer", "rewrite_question"]:
    question = get_last_question(state["messages"])
    context = state["messages"][-1].content
    prompt = GRADE_PROMPT.format(question=question, context=context)
    response = model.with_structured_output(GradeDocument).invoke(
        [{"role": "user", "content": prompt}]
    )
    if response.binary_score == "yes":
        return "generate_answer"
    else:
        return "rewrite_question"

REWRITE_PROMPT = (
    "Look at the input and try to reason about the underlying semantic intent/meaning. \n"
    "Here is the initial question:"
    "\n ---------- \n"
    "{question}"
    "\n ---------- \n"
    "Formulate an improved question:"
)

def rewrite_question(state: MessagesState):
    question = get_last_question(state["messages"])
    user_id = "user1"
    memory_context = search_memory(question, user_id)
    prompt = REWRITE_PROMPT.format(question=question)
    if memory_context:
        prompt += f"\n\nPrevious conversation context: \n{memory_context}"
    
    response = model.invoke([{"role": "user", "content": prompt}])
    return {"messages": [response]}

GENERATE_PROMPT = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you do not know the answer, just say that you do not know. "
    "Use three sentences maximum and keep the answer concise. \n"
    "Question: {question} \n"
    "Context: {context}"
)

def generate_answer(state: MessagesState):
    """Generate the answer."""
    question = get_last_question(state["messages"])
    context = state["messages"][-1].content
    user_id = "user1"

    # Search for relevant memories
    memory_context = search_memory(question, user_id)

    prompt = GENERATE_PROMPT.format(question=question, context=context)
    if memory_context:
        prompt += f"\n\nPrevious conversation context:\n{memory_context}"
    response = model.invoke([{"role": "user", "content": prompt}])
    
    # Save this interaction to memory
    save_memory(
        [
            {"role": "user", "content": question},
            {"role": "assistant", "content": response.content}
        ],
        user_id
    )
  
    return {"messages": [response]}

workflow = StateGraph(MessagesState)

workflow.add_node(generate_query_or_respond)
workflow.add_node("retrieve", ToolNode([retrieve_docs]))
workflow.add_node(rewrite_question)
workflow.add_node(generate_answer)

workflow.add_edge(START, "generate_query_or_respond")

workflow.add_conditional_edges(
    "generate_query_or_respond",
    tools_condition,
    {
        "tools": "retrieve",
        END: END
    }
)

workflow.add_conditional_edges("retrieve", grade_documents)

workflow.add_edge("generate_answer", END)
workflow.add_edge("rewrite_question", "generate_query_or_respond")

graph = workflow.compile(checkpointer=MemorySaver())

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "test"}}
    for chunk in graph.stream(
        {"messages": [{"role": "user", "content": "How to configure SR-IOV?"}]},
        config
    ):
        for node, update in chunk.items():
            print("Update from node:", node)
            update["messages"][-1].pretty_print()
            print("\n")
