import os 
import getpass
from typing import Annotated, Dict, List, Any
from typing_extensions import TypedDict, Literal

def setup_environment():
    """Setup your API KEYs"""
    if not os.environ.get("OPENAI_API_KEY"):
        openai_key = getpass.getpass("Enter your OPENAI API KEY")
        os.environ["OPENAI_API_KEY"] = openai_key
    if not os.environ.get("SERPER_API_KEY"):
        serper_key = getpass.getpass("Enter your SERPER API KEY")
        os.environ["SERPER_API_KEY"] = serper_key
    if not os.environ.get("TAVILY_API_KEY"):
        tavily_key = getpass.getpass("Enter your TAVILY API KEY")
        os.environ["TAVILY_API_KEY"] = tavily_key   

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
import operator
from langchain_tavily import TavilySearch
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver



# class State(TypedDict):
#     # messages will store our conversation history
#     ### add_messages is a special function  that append new message instead of replacing them

#     messages: Annotated[list, add_messages]


def get_llm():
    return ChatOpenAI(
        model = "gpt-4o",
        temperature= 0.0
    )



def extract_result_from_tags(tag: str, result: str):
    if "</think>" in result:
        result = result.split("</think>")[1]
    #Extract using tags

    if f"<{tag}>" in result:
        return result.split(f"<{tag}>")[1].split(f"</{tag}>")[0]

    return result


def get_sources_from_search_results(search_results):
    """Extract source URLs from Tavily search results"""
    sources = []
    try:
        # Tavily returns a list of dictionaries with 'url' keys
        if isinstance(search_results, list):
            for result in search_results:
                if isinstance(result, dict) and 'url' in result:
                    sources.append(result['url'])
        # Sometimes it might return a dict with a list under a key like 'results'
        elif isinstance(search_results, dict):
            if 'results' in search_results and isinstance(search_results['results'], list):
                for result in search_results['results']:
                    if isinstance(result, dict) and 'url' in result:
                        sources.append(result['url'])
    except Exception as e:
        print(f"Error extracting sources: {e}")
    
    return sources


class InputState(TypedDict):
    topic: str
    max_web_searchs: int


class OutputState(TypedDict):
    summary: str


class ResearchState(InputState, OutputState):
    search_query: str
    research_loop_count: int
    sources_gathered: Annotated[list, operator.add]
    web_search_results: Annotated[list, operator.add]


def generate_first_query(state: InputState):
    prompt = (
        f"Your goal is to generate a targeted web search query. ... The topic is: {state['topic']} ..."
        "Please return the query wrapped in <query> tags. For example: <query>Suggested query</query>"
    )

    reasoner_llm = get_llm()  
    # Use the invoke method with a properly formatted HumanMessage
    response = reasoner_llm.invoke([HumanMessage(content=prompt)])
  
    #Extract using tags
    query = extract_result_from_tags("query", response.content) 
    return {"search_query": query}

def web_search_generator(state: ResearchState):
    # Initialize TavilySearch first, then call it with the query
    search_tool = TavilySearch()
    search_results = search_tool.invoke(state['search_query'])

    return {
        "sources_gathered": [get_sources_from_search_results(search_results)],
        "web_search_results": [search_results],
        "research_loop_count": state["research_loop_count"] + 1 if "research_loop_count" in state else 1
    }



def summarize_sources(state: ResearchState):
    existing_summary = state.get("summary", "")
    last_web_search = state["web_search_results"][-1]
    prompt = (
        f"Generate a high-quality summary of the web search results ... The topic is: {state['topic']} ..."
        f"{f'Existing summary: {existing_summary}' if existing_summary else ''}"
        f"Search results: {last_web_search}"
        "Please return the summary wrapped in <summary> tags. For example: <summary>Suggested summary</summary>"
    )

    reasoner_llm = get_llm()
    # Use the invoke method with a properly formatted HumanMessage
    response = reasoner_llm.invoke([HumanMessage(content=prompt)])

    #Extract using tags
    summary = extract_result_from_tags("summary", response.content) 
    return {"summary": summary}



def reflect_on_suumary(state: ResearchState):
    prompt = (
        f"Identify knowledge gaps or areas that need deeper exploration ... The topic is: {state['topic']} ..."
        f"The summary is: {state['summary']}"
        "Please return the query wrapped in <query> tags. For example: <query>Suggested query</query>"
    )

    reasoner_llm = get_llm()
    # Use the invoke method with a properly formatted HumanMessage
    response = reasoner_llm.invoke([HumanMessage(content=prompt)])

    #Extract using tags
    query = extract_result_from_tags("query", response.content)    
    return {"search_query": query}



def finalize_summary(state: ResearchState):
  # Flatten the list of lists and ensure they're all strings
  all_sources_flat = []
  for source_list in state["sources_gathered"]:
    if isinstance(source_list, list):
      all_sources_flat.extend(source_list)
    else:
      all_sources_flat.append(source_list)
      
  # Remove duplicates while preserving order
  unique_sources = []
  for source in all_sources_flat:
    if source not in unique_sources and isinstance(source, str):
      unique_sources.append(source)
      
  all_sources = "\n".join(source for source in unique_sources)
  final_summary = f"## Summary\n\n{state['summary']}\n\n ### Sources:\n{all_sources}"
    
  return {"summary": final_summary}


def reasearch_router(state: ResearchState) -> Literal["finalize_summary", "web_research"]:
    if state["research_loop_count"] < state["max_web_searchs"]:
        return "web_research"
    else:
        return "finalize_summary"

def get_workflow():
    # Use input_schema and output_schema instead of the deprecated input and output parameters
    builder = StateGraph(ResearchState, input_schema=InputState, output_schema=OutputState)

    # Add nodes
    builder.add_node("generate_first_query", generate_first_query)
    builder.add_node("web_research", web_search_generator)
    builder.add_node("summarize_sources", summarize_sources)
    builder.add_node("reflect_on_summary", reflect_on_suumary)
    builder.add_node("finalize_summary", finalize_summary)

    # Add edges
    builder.add_edge(START, "generate_first_query")
    builder.add_edge("generate_first_query", "web_research")
    builder.add_edge("web_research", "summarize_sources")
    builder.add_edge("summarize_sources", "reflect_on_summary")
    builder.add_conditional_edges("reflect_on_summary", reasearch_router)
    builder.add_edge("finalize_summary", END)

    return builder.compile()


workflow = get_workflow()

result = workflow.invoke({
    "topic": "How was deepseek r1 trained",
    "max_web_searchs": 3
})


print("\n\nRESEARCH SUMMARY:\n=================\n")
print(result["summary"]) #Display the summary