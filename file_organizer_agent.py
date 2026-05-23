from dotenv import load_dotenv
from pathlib import Path
import os
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langgraph.prebuilt import ToolNode, tools_condition
import json
from typing import Literal
from pydantic import BaseModel, Field

load_dotenv()

ROOT = Path(os.getenv("ORGANIZER_ROOT", "./sample_files")).resolve()

def safe_path(relative_path: str) -> Path:
    candidate = (ROOT / relative_path).resolve()

    if candidate != ROOT and ROOT not in candidate.parents:
        raise ValueError(f"Path escapes organizer root: {relative_path}")

    return candidate

@tool
def list_files(relative_dir: str = ".", max_results: int = 50) -> str:
    """List files under a directory inside the organizer root."""
    directory = safe_path(relative_dir)

    if not directory.exists():
        return f"Directory doesn't exist: {relative_dir}"
    
    if not directory.is_dir():
        return f"Not a directory: {relative_dir}"
    

    results = []
    for path in sorted(directory.rglob("*")):
        if len(results) >= max_results:
            break

        if path.is_file():
            stat = path.stat()
            results.append({
                "path": str(path.relative_to(ROOT)),
                "size_bytes": stat.st_size,
                "extension": path.suffix.lower() or "(none)",
            })
    
    if not results:
        return "No files found."
    
    return "\n".join(str(item) for item in results)

@tool
def inspect_file(relative_path: str) -> str:
    """Inspect safe metadata for a single file inside the organizer root."""
    path = safe_path(relative_path)

    if not path.exists():
        return f"File does not exist: {relative_path}"

    if not path.is_file():
        return f"Not a file: {relative_path}"

    stat = path.stat()

    return str({
        "path": str(path.relative_to(ROOT)),
        "name": path.name,
        "stem": path.stem,
        "extension": path.suffix.lower() or "(none)",
        "size_bytes": stat.st_size,
        "parent": str(path.parent.relative_to(ROOT)),
    })

tools = [list_files, inspect_file]

SYSTEM_PROMPT = f"""
You are a local file organizer agent.

You can inspect files only inside this root:
{ROOT}

Current phase: read-only intelligence.

Rules:
- You may list files and inspect file metadata.
- You must not claim to move, rename, delete, or modify files.
- You should suggest an organization plan using clear categories.
- If the user asks you to mutate files, explain that this phase is read-only.
- Prefer using tools before making claims about the filesystem.
"""


llm = ChatOpenAI(
    model=os.getenv("LMSTUDIO_MODEL", "local-model"),
    base_url=os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1"),
    api_key=os.getenv("LMSTUDIO_API_KEY", "lm-studio"),
    temperature=0,
    timeout=60,
    max_tokens=800
)

llm_with_tools = llm.bind_tools(tools)

def assistant_node(state: MessagesState):
    print("\n[debug] calling LM Studio...", flush=True)
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = llm_with_tools.invoke(messages)
    print("[debug] LM Studio returned", flush=True)
    return {"messages": [response]}

builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant_node)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")

graph = builder.compile()

def print_trace(message):
    if isinstance(message, HumanMessage):
        print("\n[human]")
        print(message.content)

    elif isinstance(message, AIMessage):
        print("\n[assistant]")
        if message.content:
            print(message.content)

        if message.tool_calls:
            print("[tool_calls]")
            for call in message.tool_calls:
                print({
                    "name": call["name"],
                    "args": call["args"],
                    "id": call["id"],
                })

    elif isinstance(message, ToolMessage):
        print("\n[tool_result]")
        print({
            "tool_call_id": message.tool_call_id,
            "name": message.name,
            "content": message.content,
        })

    else:
        print("\n[message]")
        print(message)

def run_agent(user_input: str):
    print(f"\nOrganizer root: {ROOT}")
    events = graph.stream(
        {"messages": [HumanMessage(content=user_input)]},
        stream_mode="values"
    )

    last_message = None
    for event in events:
        last_message = event["messages"][-1]
        print_trace(last_message)

if __name__ == "__main__":
    run_agent(
        "Scan my files and suggest a simple folder organization. "
        "Do not move anything."
    )