import logging
import os
from src.graph import app
from langchain_core.messages import HumanMessage

logging.basicConfig(level=logging.ERROR)

q = "签合同时写的是“订金”，现在商家违约，我能要求双倍返还吗？"
inputs = {"messages": [HumanMessage(content=q)]}
config = {"configurable": {"thread_id": "test-trace-2"}}

print("="*60)
print(f"User Query: {q}")
for step in app.stream(inputs, config, stream_mode="updates"):
    print("-"*60)
    for node_name, state_update in step.items():
        print(f"Node Executed: {node_name}")
        if "messages" in state_update:
            for m in state_update["messages"]:
                msg_content = getattr(m, "content", str(m))
                print(f"  + Added Message: {msg_content}...\n")
        if "results_rerank" in state_update:
            docs = state_update["results_rerank"]
            print(f"  + Retrieved Contexts: {len(docs)} docs")
        if "next_agent" in state_update:
            print(f"  + Next Agent from Router: {state_update['next_agent']}")
            
print("="*60)
print("Execution Finished!")
