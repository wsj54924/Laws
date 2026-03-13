from langchain_openai import ChatOpenAI
import os
from . import agents
import operator
from langchain_core.messages import BaseMessage, HumanMessage
from dotenv import load_dotenv
from typing import Annotated,TypedDict,Sequence
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph,END
checkpointer=InMemorySaver()
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_BASE_URL"] = os.getenv("OPENAI_BASE_URL")
class AgentState(TypedDict):
    messages:Annotated[Sequence[BaseMessage],operator.add]
    results_rerank:str
    next_agent:str|None
    legal_consultant_result: BaseMessage | None
llm=ChatOpenAI(model="Qwen/Qwen3-32B", streaming=True)


def Router_node(state: AgentState):
    messages = state.get('messages', [])
    legal_consultant_result = state.get('legal_consultant_result')

    # If the last message is from Auditor, use deterministic routing
    if messages:
        last_msg = messages[-1]
        last_content = getattr(last_msg, 'content', str(last_msg))
        if "[RETRY_CONSULTANT]" in last_content:
            return {"next_agent": "Legal_Consultant"}
        elif "[RETRY_RESEARCH]" in last_content:
            return {"next_agent": "Researcher"}
        elif "[APPROVE]" in last_content:
            messages_to_return = []
            if legal_consultant_result is not None:
                messages_to_return.append(legal_consultant_result)
            return {
                "next_agent": "FINISH",
                "messages": messages_to_return
            }

    router = agents.Router()
    response = router.invoke(messages).content.strip()

    if response not in ["query_rewrite", "Researcher", "Legal_Consultant", "Auditor"]:
        # Router 返回了直接回答，需要返回给用户
        messages_to_return = []
        # 如果有 legal_consultant_result，也一并返回
        if legal_consultant_result is not None:
            messages_to_return.append(legal_consultant_result)
        # 添加 Router 的回答
        messages_to_return.append(response)
        return {
            "next_agent": "FINISH",
            "messages": messages_to_return
        }
    return {"next_agent": response}
def query_rewrite_node(state:AgentState):
    message=state['messages'][-1].content
    query_rewrite_agent=agents.query_rewrite()
    result=query_rewrite_agent.invoke([message])
    return {"messages":[result]}
def Researcher_node(state:AgentState):
    messages=state['messages']
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    db_dir = os.path.join(parent_dir, "chroma_db")
    chain=agents.Researcher_Agent(persist_directory=db_dir)
    result=chain.invoke(messages)
    return {"results_rerank": result}
def Legal_Consultant_node(state:AgentState):
    results_rerank=state['results_rerank']
    # Get the last HumanMessage (user's latest question)
    messages=state['messages']
    
    # Get Auditor's check result if available
    Auditor_result = None
    if len(messages) >= 2 and hasattr(messages[-2], 'content'):
        Auditor_result = messages[-2].content
    
    # Find the user's question (last HumanMessage)
    query = None
    for msg in reversed(messages):
        if hasattr(msg, 'type') and msg.type == 'human':
            query = msg
            break
    
    if query is None:
        query = messages[-1]
    
    chain=agents.Legal_Consultant_Agent(results_rerank,Auditor_result)
    result=chain.invoke(query)
    return {"messages":[result] ,"legal_consultant_result": result }
def Auditor_node(state:AgentState):
    results_rerank=state['results_rerank']
    legal_consultant_result = state['legal_consultant_result']
    # Get the last AIMessage (Legal Consultant's response)
    query = legal_consultant_result
    chain=agents.Auditor_Agent(results_rerank)
    result=chain.invoke(query)
    return {"messages":[result]}

workflow=StateGraph(AgentState)
workflow.add_node("Router_node",Router_node)
workflow.add_node("query_rewrite_node",query_rewrite_node)
workflow.add_node("Researcher_node",Researcher_node)
workflow.add_node("Legal_Consultant_node",Legal_Consultant_node)
workflow.add_node("Auditor_node",Auditor_node)

workflow.set_entry_point("Router_node")

# Routing Logic
workflow.add_conditional_edges(
    "Router_node",
    lambda x:x["next_agent"],
    {
        "query_rewrite":"query_rewrite_node",
        "Researcher":"Researcher_node",
        "Legal_Consultant":"Legal_Consultant_node",
        "Auditor":"Auditor_node",
        "FINISH":END
    }
)


# Standard Sequential Workflow from query_rewrite -> Researcher -> Consultant -> Auditor -> END
workflow.add_edge("query_rewrite_node", "Researcher_node")
workflow.add_edge("Researcher_node", "Legal_Consultant_node")
workflow.add_edge("Legal_Consultant_node", "Auditor_node")
workflow.add_edge("Auditor_node", "Router_node")

app=workflow.compile(checkpointer=checkpointer)
