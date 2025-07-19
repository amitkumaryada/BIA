"""
Multi-Tool Research Assistant (AutoGen Version)

An AutoGen-based agent that replicates the LangGraph research assistant functionality:
1. Generate targeted search queries for a given topic
2. Search the web for information using Tavily
3. Summarize search results
4. Reflect on summaries to identify knowledge gaps
5. Generate additional search queries to fill gaps
6. Create comprehensive research reports with sources
"""

import os
import asyncio
from typing import Annotated
from dotenv import load_dotenv
import json

# Import AutoGen components
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient

# Import Tavily for web search
from tavily import TavilyClient

# Load environment variables
load_dotenv()

# Initialize the Tavily client
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# Global research state to track progress
research_state = {
    "topic": "",
    "max_searches": 3,
    "current_searches": 0,
    "sources": [],
    "summaries": [],
    "final_report": ""
}

# Tool functions for the research assistant
def web_search(query: Annotated[str, "The search query to find information about"]) -> str:
    """Search the web for information using Tavily API.
    
    Args:
        query: The search query string
        
    Returns:
        JSON string containing search results
    """
    try:
        # Perform the search
        search_results = tavily_client.search(query=query, search_depth="advanced", max_results=5)
        
        # Update research state
        research_state["current_searches"] += 1
        
        # Extract sources
        sources = []
        if isinstance(search_results, dict) and 'results' in search_results:
            for result in search_results['results']:
                if 'url' in result:
                    sources.append(result['url'])
        
        # Store sources
        research_state["sources"].extend(sources)
        
        # Return formatted results
        return json.dumps({
            "query": query,
            "results": search_results,
            "sources_found": len(sources),
            "total_searches": research_state["current_searches"]
        }, indent=2)
        
    except Exception as e:
        return f"Error performing web search: {str(e)}"

def get_research_status() -> str:
    """Get the current status of the research process.
    
    Returns:
        Current research progress and statistics
    """
    return json.dumps({
        "topic": research_state["topic"],
        "searches_completed": research_state["current_searches"],
        "max_searches": research_state["max_searches"],
        "sources_gathered": len(research_state["sources"]),
        "summaries_created": len(research_state["summaries"]),
        "can_continue_searching": research_state["current_searches"] < research_state["max_searches"]
    }, indent=2)

def save_summary(summary: Annotated[str, "The summary text to save"]) -> str:
    """Save a research summary.
    
    Args:
        summary: The summary text to save
        
    Returns:
        Confirmation message
    """
    research_state["summaries"].append(summary)
    return f"Summary saved. Total summaries: {len(research_state['summaries'])}"

def create_final_report() -> str:
    """Create the final research report with all summaries and sources.
    
    Returns:
        The complete research report
    """
    # Create the final report
    report = f"# Research Report: {research_state['topic']}\n\n"
    
    # Add summaries
    if research_state["summaries"]:
        report += "## Summary\n\n"
        for i, summary in enumerate(research_state["summaries"], 1):
            report += f"### Section {i}\n{summary}\n\n"
    
    # Add sources
    if research_state["sources"]:
        report += "## Sources\n\n"
        unique_sources = list(set(research_state["sources"]))  # Remove duplicates
        for i, source in enumerate(unique_sources, 1):
            report += f"{i}. {source}\n"
    
    # Add research statistics
    report += f"\n## Research Statistics\n\n"
    report += f"- Searches performed: {research_state['current_searches']}\n"
    report += f"- Sources found: {len(research_state['sources'])}\n"
    report += f"- Summaries created: {len(research_state['summaries'])}\n"
    
    # Store the final report
    research_state["final_report"] = report
    
    return report

async def main(topic: str = "How was DeepSeek R1 trained", max_searches: int = 3):
    """Main function to run the research assistant."""
    
    # Check for required API keys
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please set the OPENAI_API_KEY environment variable.")
    
    tavily_key = os.getenv("TAVILY_API_KEY")
    if not tavily_key:
        raise ValueError("Please set the TAVILY_API_KEY environment variable.")
    
    # Initialize research state
    research_state["topic"] = topic
    research_state["max_searches"] = max_searches
    research_state["current_searches"] = 0
    research_state["sources"] = []
    research_state["summaries"] = []
    research_state["final_report"] = ""
    
    # Initialize model client
    model_client = OpenAIChatCompletionClient(model="gpt-4o", temperature=0.2)
    
    try:
        # Create the research assistant
        assistant = AssistantAgent(
            name="research_assistant",
            model_client=model_client,
            system_message=(
                f"You are an expert research assistant investigating: '{topic}'\n\n"
                "Your research process:\n"
                "1. Use web_search to find information on the topic\n"
                "2. Analyze the search results and save_summary of key findings\n"
                "3. Use get_research_status to check your progress\n"
                "4. Continue searching for different aspects until you reach the search limit\n"
                "5. Use create_final_report to generate the comprehensive report\n\n"
                f"Important guidelines:\n"
                f"- You can perform up to {max_searches} searches\n"
                "- Focus on finding accurate, comprehensive information\n"
                "- Save summaries after analyzing each search\n"
                "- When you complete your research, say 'RESEARCH COMPLETE'\n"
                "- Always include sources in your final report"
            ),
            tools=[web_search, get_research_status, save_summary, create_final_report]
        )
        
        # Create user proxy that doesn't require human input
        user_proxy = UserProxyAgent(
            name="user_proxy",
            input_func=lambda _: ""  # No human input required
        )
        
        # Define termination condition
        termination_condition = TextMentionTermination("RESEARCH COMPLETE")
        
        # Create the team
        team = RoundRobinGroupChat(
            [assistant, user_proxy],
            termination_condition=termination_condition
        )
        
        # Start the research process
        print(f"\n🔍 Starting research on: {topic}")
        print(f"📊 Maximum searches allowed: {max_searches}")
        print("=" * 60)
        
        # Run the conversation with streaming
        stream = team.run_stream(task=f"Research the topic: {topic}")
        await Console(stream)
        
        # Display final results
        print("\n" + "=" * 60)
        print("🎯 RESEARCH COMPLETED")
        print("=" * 60)
        
        if research_state["final_report"]:
            print("\n📋 FINAL RESEARCH REPORT:")
            print("-" * 40)
            print(research_state["final_report"])
        else:
            print("\n⚠️  No final report was generated.")
            
    finally:
        # Ensure the model client is closed properly
        await model_client.close()

if __name__ == "__main__":
    import sys
    
    # Get the research topic from command line arguments or use a default
    if len(sys.argv) > 1:
        topic = sys.argv[1]
    else:
        topic = "How was DeepSeek R1 trained"
    
    # Get the maximum number of searches
    max_searches = 3
    if len(sys.argv) > 2:
        try:
            max_searches = int(sys.argv[2])
        except ValueError:
            print(f"Invalid max_searches value. Using default: {max_searches}")
    
    # Run the research assistant
    asyncio.run(main(topic, max_searches))