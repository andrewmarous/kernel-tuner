import os
import json
from openai.types.chat import ChatCompletion
from typing import TypedDict

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel
from langgraph.graph import StateGraph, END
from openai import OpenAI

from tools import KernelTuner


SYSTEM_PROMPT = """
You are a CUDA Performance Engineering Agent. Your goal is to optimize a Matrix Multiplication kernel.
You have access to a tool `update_and_run_kernel(x, y)` which compiles the code and returns Nsight Compute metrics.

### Optimization Logic:
1. If 'sm__throughput' is low (< 30%), the GPU is underutilized.
2. If 'sm__warps_active.avg.pct_of_peak' (Occupancy) is low (< 50%), your block size might be too large,
   causing register pressure or shared memory exhaustion. Try smaller block sizes.
3. If 'gpu__compute_memory_throughput' is high (> 70%) but SM throughput is low, you are Memory-bound.
   Tiling might help (though currently, you are only adjusting block sizes).
4. Matrix dimensions are often 128, 512, and 1024. Block sizes that are powers of two (8, 16, 32)
   usually perform best due to warp alignment (32 threads).

### Constraints:
- MATMUL_BLOCK_SIZE_X * MATMUL_BLOCK_SIZE_Y must be <= 1024 (Threads per block limit).
- Always explain your reasoning based on the metrics before calling the tool.
"""

if not load_dotenv():
    raise Exception("env load failed")

llm = OpenAI(
        api_key=os.getenv("INCEPTION_API_KEY"),
        base_url="https://api.inceptionlabs.ai/v1"
        )

class AgentState(TypedDict):
    current_params: dict[str, int]
    metrics: dict[str, int | float | bool]
    history: list[dict[str, int | float | bool]]
    error: str

def update_and_run_kernel():
    ...

def orchestrator_node(state: AgentState):
    messages: list = [{"role": "system", "content": SYSTEM_PROMPT}]

    if not state["history"]:
        messages.append({"role": "user", "content": "Start the optimization. Try a baseline of 16x16."})
    else:
        last_run = state["history"][-1]
        msg = f"Last Attempt: {last_run['params']}\nMetrics: {last_run['metrics']}"
        messages.append({"role": "user", "content": msg})

    tools = [
        {
            "type": "function",
            "function": {
                "name": "update_and_run_kernel",
                "description": "Compiles the CUDA kernel with the given block sizes and returns NCU metrics.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "x": {"type": "integer", "description": "Block size X (e.g., 16, 32)"},
                        "y": {"type": "integer", "description": "Block size Y (e.g., 16, 32)"}
                    },
                    "required": ["x", "y"]
                }
            }
        }
    ]

    response: ChatCompletion = llm.chat.completions.create(
        model = "mercury-2",
        messages = messages,
        tools=tools,
        tool_choice="auto"
    )

    response_message = response.choices[0].message

    return {"next_action": response_message}

def execution_node(state: AgentState):
    message = state.get("next_action")

    if not message or not message.tool_calls:
        return {"error": "Execution node called but no tool calls found."}

    tool_call = message.tool_calls[0]
    try:
        args = json.loads(tool_call.function.arguments)
        x = args.get("x", 16)
        y = args.get("y", 16)
    except json.JSONDecodeError:
        return {"error": "Failed to parse tool call arguments."}

    print(f"\n[Executor] Agent requested Block Sizes: X={x}, Y={y}")

    tuner = KernelTuner("matmul.cu")
    tuner.update_params(x, y)
    if (compilation_err := tuner.compile()):
        return {"error": compilation_err}

    current_metrics = tuner.profile_kernel()
    if "error" in current_metrics:
        return {"error": current_metrics["error"]}

    print(f"[Executor] NCU metrics retrieved: {current_metrics}")

    state["history"].append(current_metrics)
    return

def testing_node(state: AgentState):




workflow = StateGraph(AgentState)
workflow.add_node(orchestrator_node)
workflow.add_node(execution_node)
workflow.add_node(testing_node)

workflow.add_conditional_edges("orchestrator_node")

def main():
    print()


if __name__ == "__main__":
    main()
