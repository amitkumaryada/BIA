"""    Task:
    Create a Personal Assistant Agent that can:
    Have basic conversations
    Remember user's name and preferences across sessions
    Perform simple calculations
    Tell current time/date
"""

import os 
import getpass
from typing import Annotated, Dict, List, Any
from typing_extensions import TypedDict

def setup_environment():
    """Setup your API KEYs"""
    if not os.environ.get("OPENAI_API_KEY"):
        openai_key = getpass.getpass("Enter your OPENAI API KEY")
        os.environ["OPENAI_API_KEY"] = openai_key

    if not os.environ.get("LANGSMITH_API_KEY"):
        langsmith_key = getpass.getpass("ENTER your LAngsmith API KEY")
        if langsmith_key:
            os.environ["LANGSMITH_API_KEY"] = langsmith_key
            os.environ["LANGCHAIN_TRACING_V2"] ="true"
            os.environ["LANGCHAIN_PROJECT"] = "Langgraph tutorial"
        else:
            print("skipping langsmith set up")


    print("Envioenment setup completed")

setup_environment()



from langgraph.graph import StateGraph , START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
from typing import TypedDict, Annotated 
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
import datetime
import math


class State(TypedDict):
    # messages will store our conversation history
    ### add_messages is a special function  that append new message instead of replacing them

    messages: Annotated[list, add_messages]




llm = ChatOpenAI(
    model = "gpt-4o",
    temperature= 0.0
)

llm.invoke("Hi").content


@tool
def get_current_time() -> str:
    """Get the current time and date."""
    return f"Current time is {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

@tool
def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression safely.
    
    Args:
        expression: A mathematical expression to evaluate (e.g., "2 + 3 * 4")
        
    Returns:
        The result of the calculation
    """
    try:
        # Safe evaluation of mathematical expressions
        # Only allow basic math operations
        allowed_names = {
            k: v for k, v in math.__dict__.items() if not k.startswith("__")
        }
        allowed_names.update({"abs": abs, "round": round})
        
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return f"The result of {expression} is {result}"
    except Exception as e:
        return f"Error calculating {expression}: {str(e)}"


tools = [get_current_time, calculator]

llm_with_tools = llm.bind_tools(tools)

def chatbot(state):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


  

# Initialize memory checkpoint
checkpointer = InMemorySaver()
# memory = MemorySaver(State)
graph_builder = StateGraph(State)
graph_builder.add_node("chat", chatbot)
tools = ToolNode(tools)
graph_builder.add_node("tools", tools)
graph_builder.add_edge(START, "chat")
graph_builder.add_conditional_edges("chat", tools_condition)
graph_builder.add_edge("tools","chat")
graph = graph_builder.compile(checkpointer=checkpointer)



from IPython.display import Image, display

try:
    display(Image(graph.get_graph().draw_mermaid_png()))
except Exception:
    # This requires some extra dependencies and is optional
    pass



config = {"configurable": {"thread_id": "1"}}
result = graph.invoke({"messages":[{"role":'user',"content":"Hello! my name is amit I take lectures on BIA"}]}, config)
print(result["messages"][-1].content)
