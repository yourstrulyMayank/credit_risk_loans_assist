from typing import List, Dict, TypedDict
from langchain_core.documents import Document
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langgraph.graph import StateGraph, END
from langchain_ollama import OllamaLLM
from logger_utils import setup_logger

logger = setup_logger()


# ---- Shared State ----
class State(TypedDict):
    docs: List[Document]
    accumulated_summaries: List[str]
    current_summary: str
    final_summary: str


# ---- LLM Setup ----
llm = OllamaLLM(model="llama3.2")

map_prompt = PromptTemplate.from_template(
    "Write an excellent summary of the following, covering every critical point:\n\n{context}"
)
map_chain = map_prompt | llm

reduce_prompt = PromptTemplate.from_template(
    "The following is a set of summaries:\n{summaries}\nTake these and distill it into a final, consolidated summary of the main themes."
)
reduce_chain = reduce_prompt | llm


# ---- LangGraph Nodes ----
def map_node(state: State) -> State:
    doc = state["docs"].pop(0)
    result = map_chain.invoke({"context": doc.page_content})
    result_text = result if isinstance(result, str) else result.get("text", "")
    state["accumulated_summaries"].append(result_text["text"])
    return state


def reduce_node(state: State) -> State:
    combined = "\n".join(state["accumulated_summaries"])
    result = reduce_chain.invoke({"summaries": combined})
    result_text = result if isinstance(result, str) else result.get("text", "")
    state["final_summary"] = result_text["text"]
    return state


def should_continue(state: State) -> str:
    return "map" if len(state["docs"]) > 0 else "reduce"


# ---- Main Entrypoint ----
def generate_summary(docs: List[Document], file_name: str = "") -> str:
    """
    Generate a summary from a list of Document objects using LangGraph map-reduce.
    """
    if not docs:
        logger.warning(f"No documents found for summarization of file: {file_name}")
        return "No content found for summary."

    logger.info(f"Running map-reduce summarization on {len(docs)} docs from: {file_name}")

    initial_state: State = {
        "docs": docs.copy(),
        "accumulated_summaries": [],
        "current_summary": "",
        "final_summary": "",
    }

    graph_builder = StateGraph(State)
    graph_builder.add_node("map", map_node)
    graph_builder.add_node("reduce", reduce_node)
    graph_builder.set_conditional_entry_point(should_continue)
    graph_builder.add_edge("map", "reduce")
    graph_builder.add_edge("reduce", END)
    graph = graph_builder.compile()

    final_state = graph.invoke(initial_state)
    summary = final_state.get("final_summary", "No summary generated.").strip()

    logger.info(f"Summary generated for {file_name}: {summary[:100]}...")
    return summary
