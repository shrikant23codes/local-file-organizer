from dotenv import load_dotenv
from pathlib import Path
import os
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langgraph.prebuilt import ToolNode, tools_condition
import json
from typing import Literal, List
from pydantic import BaseModel, Field

load_dotenv()

ROOT = Path(os.getenv("ORGANIZER_ROOT", "./sample_files")).resolve()

def safe_path(relative_path: str) -> Path:
    candidate = (ROOT / relative_path).resolve()

    if candidate != ROOT and ROOT not in candidate.parents:
        raise ValueError(f"Path escapes organizer root: {relative_path}")

    return candidate


class FilePlanAction(BaseModel):
    operation: Literal["move", "rename", "keep"] = Field(
        description="The proposed operation. This phase only plans; it never executes."
    )
    source_path: str = Field(description="Existing file path relative to ORGANIZER_ROOT.")
    destination_path: str = Field(description="Proposed destination path relative to ORGANIZER_ROOT.")
    category: str = Field(description="Human-readable category, such as Documents, Images, Notes.")
    reason: str = Field(description="Short explanation for why this action is useful.")
    confidence: float = Field(ge=0, le=1, description="Confidence from 0 to 1.")

class OrganizationPlan(BaseModel):
    summary: str = Field(description="Short summary of the proposed organization.")
    actions: list[FilePlanAction] = Field(description="Proposed file organization actions.")
    assumptions: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)

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


@tool(args_schema=OrganizationPlan)
def propose_organization_plan(
    summary: str,
    actions: List[FilePlanAction],
    assumptions: List[str] | None = None,
    risks: List[str] | None = None
) -> str:
    """Submit a structured read-only organization plan. This does not move, rename, or modify files."""
    assumptions = assumptions or []
    risks = risks or []

    validation_errors = []
    for index, action in enumerate(actions):
        try:
            source = safe_path(action.source_path)
            destination = safe_path(action.destination_path)
            
            if not source.exists():
                validation_errors.append(f"Action {index}: source does not exist: {action.source_path}")
            
            if action.operation in ("move", "rename") and source == destination:
                validation_errors.append(f"Action {index}: move/rename destination is same as source: {action.source_path}")
            
            if action.operation == "keep" and source != destination:
                validation_errors.append(
                    f"Action {index}: operation='keep' requires source_path and destination_path "
                    f"to be identical. Use operation='move' when proposing a new folder."
                )            
        except Exception as e:
            validation_errors.append(f"Action {index}: invalid path: {e}")
    
    plan = {
        "summary": summary,
        "actions": [action.model_dump() for action in actions],
        "assumptions": assumptions,
        "risks": risks,
        "validation_errors": validation_errors,
        "executable": False,
        "phase": "planning_only",
    }

    return json.dumps(plan, indent=2)

tools = [list_files, inspect_file, propose_organization_plan]

SYSTEM_PROMPT = f"""
Your goal is to propose a better organization, not merely categorize files in place.
Do not use hidden reasoning or thinking mode. Respond directly with either a tool call or a concise final answer.


You can inspect files only inside this root:
{ROOT}

Current phase: structured planning only.
If a file is currently at the root and belongs in a category folder, propose operation="move" with a destination_path like:
- Documents/resume.pdf
- Images/vacation.jpg
- Notes/notes.txt

Rules:
- You may list files and inspect file metadata.
- You must propose organization using the propose_organization_plan tool.
- You must not move, rename, delete, create, or modify files.
- Every proposed source_path and destination_path must be relative to the organizer root.
- Prefer simple destination folders like Documents/, Images/, Notes/, Archives/, Code/, Misc/.
- Use operation="keep" only when the file is already in the right destination folder and source_path equals destination_path.
- If destination_path is different from source_path, use operation="move" or "rename".- If uncertain, include the uncertainty in assumptions or risks.
- After the plan tool returns, summarize the plan clearly for the user.

Examples:
- keep: source_path="notes.txt", destination_path="notes.txt"
- move: source_path="notes.txt", destination_path="Notes/notes.txt"
- rename: source_path="tax_2024.pdf", destination_path="tax-return-2024.pdf"
"""

config = {
    "run_name": "phase-2-organization-plan",
    "tags": ["local-file-organizer", "phase-2", "planning-only"],
    "metadata": {
        "phase": "phase-2",
        "agent": "local-file-organizer",
        "model": os.getenv("LMSTUDIO_MODEL", "local-model"),
        "organizer_root": str(ROOT),
    },
}

llm = ChatOpenAI(
    model=os.getenv("LMSTUDIO_MODEL", "local-model"),
    base_url=os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1"),
    api_key=os.getenv("LMSTUDIO_API_KEY", "lm-studio"),
    temperature=0,
    timeout=120,
    max_tokens=2000
)

llm_with_tools = llm.bind_tools(tools)

def assistant_node(state: MessagesState):
    print("\n[debug] calling LM Studio...", flush=True)
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = llm_with_tools.invoke(messages)
    print(f"[debug] LM Studio returned {response}", flush=True)
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
        config=config,
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