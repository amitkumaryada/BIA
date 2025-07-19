#!/usr/bin/env python3
"""
Research Assistant using CrewAI Framework
Replicates the original LangGraph research assistant functionality with multi-agent workflow
"""

import os
import getpass
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime
import json

# CrewAI imports
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

# Environment setup
def setup_environment():
    """Setup API keys for OpenAI, Tavily, and LangSmith"""
    if not os.environ.get("OPENAI_API_KEY"):
        openai_key = getpass.getpass("Enter your OPENAI API KEY: ")
        os.environ["OPENAI_API_KEY"] = openai_key
    
    if not os.environ.get("TAVILY_API_KEY"):
        tavily_key = getpass.getpass("Enter your TAVILY API KEY: ")
        os.environ["TAVILY_API_KEY"] = tavily_key
    
    if not os.environ.get("LANGSMITH_API_KEY"):
        langsmith_key = getpass.getpass("ENTER your LangSmith API KEY (optional): ")
        if langsmith_key:
            os.environ["LANGSMITH_API_KEY"] = langsmith_key
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_PROJECT"] = "CrewAI Research Assistant"
        else:
            print("Skipping LangSmith setup")
    
    print("Environment setup completed")

setup_environment()

# Data Models
class ResearchState(BaseModel):
    """Global state for research workflow"""
    topic: str = ""
    max_web_searches: int = 3
    current_search_count: int = 0
    search_queries: List[str] = Field(default_factory=list)
    web_search_results: List[Dict[str, Any]] = Field(default_factory=list)
    sources_gathered: List[str] = Field(default_factory=list)
    summaries: List[str] = Field(default_factory=list)
    final_summary: str = ""
    research_complete: bool = False

# Global state instance
research_state = ResearchState()

# CrewAI Tools
class QueryGenerationTool(BaseTool):
    name: str = "generate_search_query"
    description: str = "Generate targeted web search queries based on research topic and existing knowledge"
    
    def _run(self, topic: str, existing_summary: str = "") -> str:
        """Generate a search query for the given topic"""
        try:
            if not existing_summary:
                # First query - broad search
                prompt = f"""
                Your goal is to generate a targeted web search query for comprehensive research.
                
                Topic: {topic}
                
                Generate a specific, focused search query that will help gather comprehensive information about this topic.
                Focus on finding authoritative sources, recent developments, and key details.
                
                Return only the search query, no additional text.
                """
            else:
                # Follow-up query - identify gaps
                prompt = f"""
                Based on the existing research summary, identify knowledge gaps and generate a targeted search query.
                
                Topic: {topic}
                Existing Summary: {existing_summary}
                
                Generate a specific search query that will help fill knowledge gaps or provide additional depth.
                Focus on areas not well covered in the existing summary.
                
                Return only the search query, no additional text.
                """
            
            llm = ChatOpenAI(model="gpt-4o", temperature=0.0)
            response = llm.invoke(prompt)
            query = response.content.strip()
            
            # Store the query
            research_state.search_queries.append(query)
            
            return query
            
        except Exception as e:
            return f"Error generating query: {str(e)}"

class WebSearchTool(BaseTool):
    name: str = "web_search"
    description: str = "Perform web search using Tavily API and gather sources"
    
    def _run(self, query: str) -> str:
        """Perform web search and return results"""
        try:
            search_tool = TavilySearch()
            search_results = search_tool.invoke(query)
            
            # Extract sources from search results
            sources = self._extract_sources(search_results)
            
            # Update global state
            research_state.web_search_results.append(search_results)
            research_state.sources_gathered.extend(sources)
            research_state.current_search_count += 1
            
            # Format results for agent
            formatted_results = self._format_search_results(search_results)
            
            return f"Search completed for query: '{query}'\n\nResults:\n{formatted_results}\n\nSources found: {len(sources)}"
            
        except Exception as e:
            return f"Error performing web search: {str(e)}"
    
    def _extract_sources(self, search_results) -> List[str]:
        """Extract source URLs from Tavily search results"""
        sources = []
        try:
            if isinstance(search_results, list):
                for result in search_results:
                    if isinstance(result, dict) and 'url' in result:
                        sources.append(result['url'])
            elif isinstance(search_results, dict):
                if 'results' in search_results and isinstance(search_results['results'], list):
                    for result in search_results['results']:
                        if isinstance(result, dict) and 'url' in result:
                            sources.append(result['url'])
        except Exception as e:
            print(f"Error extracting sources: {e}")
        
        return sources
    
    def _format_search_results(self, search_results) -> str:
        """Format search results for display"""
        try:
            if isinstance(search_results, list):
                formatted = "\n".join([f"- {result.get('title', 'No title')}: {result.get('content', 'No content')[:200]}..." 
                                     for result in search_results[:5]])
            elif isinstance(search_results, dict) and 'results' in search_results:
                results = search_results['results'][:5]
                formatted = "\n".join([f"- {result.get('title', 'No title')}: {result.get('content', 'No content')[:200]}..." 
                                     for result in results])
            else:
                formatted = str(search_results)[:500] + "..."
            
            return formatted
        except Exception as e:
            return f"Error formatting results: {str(e)}"

class SummarizationTool(BaseTool):
    name: str = "summarize_research"
    description: str = "Summarize web search results and integrate with existing knowledge"
    
    def _run(self, search_results: str, existing_summary: str = "") -> str:
        """Summarize the latest search results"""
        try:
            prompt = f"""
            Generate a comprehensive summary of the web search results for the research topic.
            
            Topic: {research_state.topic}
            
            {f'Existing Summary: {existing_summary}' if existing_summary else ''}
            
            Latest Search Results:
            {search_results}
            
            Instructions:
            1. Create a well-structured summary that integrates new information with existing knowledge
            2. Focus on key facts, findings, and insights
            3. Maintain accuracy and cite important details
            4. If this is an update to existing summary, merge information coherently
            5. Organize information logically with clear sections
            
            Return a comprehensive summary that builds upon previous knowledge.
            """
            
            llm = ChatOpenAI(model="gpt-4o", temperature=0.0)
            response = llm.invoke(prompt)
            summary = response.content.strip()
            
            # Store the summary
            research_state.summaries.append(summary)
            
            return summary
            
        except Exception as e:
            return f"Error summarizing research: {str(e)}"

class ResearchStatusTool(BaseTool):
    name: str = "get_research_status"
    description: str = "Get current research progress and determine if more searches are needed"
    
    def _run(self) -> str:
        """Get current research status"""
        try:
            status = {
                "topic": research_state.topic,
                "searches_completed": research_state.current_search_count,
                "max_searches": research_state.max_web_searches,
                "queries_used": research_state.search_queries,
                "sources_found": len(research_state.sources_gathered),
                "summaries_created": len(research_state.summaries),
                "research_complete": research_state.research_complete
            }
            
            # Determine if more research is needed
            needs_more_research = (
                research_state.current_search_count < research_state.max_web_searches and 
                not research_state.research_complete
            )
            
            status["needs_more_research"] = needs_more_research
            
            return json.dumps(status, indent=2)
            
        except Exception as e:
            return f"Error getting research status: {str(e)}"

class FinalReportTool(BaseTool):
    name: str = "create_final_report"
    description: str = "Create final comprehensive research report with sources"
    
    def _run(self) -> str:
        """Create final research report"""
        try:
            # Get the latest summary
            latest_summary = research_state.summaries[-1] if research_state.summaries else "No summary available"
            
            # Remove duplicate sources
            unique_sources = list(set(research_state.sources_gathered))
            
            # Create final report
            final_report = f"""
# Research Report: {research_state.topic}

## Executive Summary

{latest_summary}

## Research Statistics

- **Searches Performed**: {research_state.current_search_count}
- **Sources Gathered**: {len(unique_sources)}
- **Summaries Created**: {len(research_state.summaries)}
- **Research Queries**: {len(research_state.search_queries)}

## Sources

{chr(10).join([f"- {source}" for source in unique_sources])}

## Search Queries Used

{chr(10).join([f"{i+1}. {query}" for i, query in enumerate(research_state.search_queries)])}

---
*Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
            """
            
            research_state.final_summary = final_report
            research_state.research_complete = True
            
            return final_report
            
        except Exception as e:
            return f"Error creating final report: {str(e)}"

# CrewAI Agents
def create_research_coordinator():
    """Create the research coordinator agent"""
    return Agent(
        role="Research Coordinator",
        goal="Coordinate comprehensive research on given topics by planning search strategies and managing workflow",
        backstory="""You are an expert research coordinator with extensive experience in 
        conducting thorough investigations. You excel at breaking down complex topics into 
        searchable components and ensuring comprehensive coverage of all relevant aspects.""",
        tools=[QueryGenerationTool(), ResearchStatusTool()],
        llm=ChatOpenAI(model="gpt-4o", temperature=0.0),
        verbose=True,
        allow_delegation=True
    )

def create_web_researcher():
    """Create the web researcher agent"""
    return Agent(
        role="Web Research Specialist",
        goal="Perform targeted web searches and gather high-quality information from reliable sources",
        backstory="""You are a skilled web researcher with expertise in finding authoritative 
        sources and extracting valuable information. You know how to craft effective search 
        queries and evaluate source credibility.""",
        tools=[WebSearchTool(), ResearchStatusTool()],
        llm=ChatOpenAI(model="gpt-4o", temperature=0.0),
        verbose=True
    )

def create_content_analyst():
    """Create the content analyst agent"""
    return Agent(
        role="Content Analysis Specialist",
        goal="Analyze and synthesize research findings into comprehensive summaries and reports",
        backstory="""You are an expert content analyst with strong skills in synthesizing 
        information from multiple sources. You excel at creating clear, well-structured 
        summaries that capture key insights and maintain accuracy.""",
        tools=[SummarizationTool(), FinalReportTool(), ResearchStatusTool()],
        llm=ChatOpenAI(model="gpt-4o", temperature=0.0),
        verbose=True
    )

# CrewAI Tasks
def create_research_planning_task(topic: str, max_searches: int):
    """Create task for research planning"""
    return Task(
        description=f"""
        Plan and initiate comprehensive research on the topic: "{topic}"
        
        Your responsibilities:
        1. Generate an initial targeted search query for this topic
        2. Check research status to understand current progress
        3. Ensure the research strategy covers all important aspects of the topic
        
        Topic: {topic}
        Maximum searches allowed: {max_searches}
        
        Use the available tools to generate the first search query and get research status.
        """,
        expected_output="Initial search query generated and research status obtained",
        agent=create_research_coordinator()
    )

def create_web_search_task():
    """Create task for web searching"""
    return Task(
        description="""
        Perform web search using the generated query and gather relevant information.
        
        Your responsibilities:
        1. Execute the web search using the latest query
        2. Gather and organize search results
        3. Extract source URLs for reference
        4. Check research status to determine next steps
        
        Use the web search tool to find comprehensive information on the research topic.
        """,
        expected_output="Web search completed with results and sources gathered",
        agent=create_web_researcher()
    )

def create_analysis_task():
    """Create task for content analysis"""
    return Task(
        description="""
        Analyze and summarize the research findings from web search results.
        
        Your responsibilities:
        1. Summarize the latest web search results
        2. Integrate findings with any existing research
        3. Check if more research is needed based on current progress
        4. Generate follow-up queries if research should continue
        
        Create a comprehensive summary that builds upon previous knowledge.
        """,
        expected_output="Research findings summarized and integrated",
        agent=create_content_analyst()
    )

def create_final_report_task():
    """Create task for final report generation"""
    return Task(
        description="""
        Create the final comprehensive research report.
        
        Your responsibilities:
        1. Generate a final comprehensive report with all findings
        2. Include all sources and research statistics
        3. Ensure the report is well-structured and complete
        4. Mark research as complete
        
        This should be a polished, professional research report.
        """,
        expected_output="Final comprehensive research report with sources and statistics",
        agent=create_content_analyst()
    )

# Main Research Assistant Class
class ResearchAssistantCrew:
    """Main class for CrewAI Research Assistant"""
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.0)
    
    def conduct_research(self, topic: str, max_searches: int = 3) -> str:
        """Conduct comprehensive research on a topic"""
        # Reset global state
        global research_state
        research_state = ResearchState(topic=topic, max_web_searches=max_searches)
        
        print(f"\n🔍 Starting research on: {topic}")
        print(f"📊 Maximum searches: {max_searches}")
        print("=" * 60)
        
        # Continue research loop until complete
        while research_state.current_search_count < max_searches and not research_state.research_complete:
            print(f"\n🔄 Research iteration {research_state.current_search_count + 1}/{max_searches}")
            
            # Create crew for this iteration
            crew = self._create_research_crew()
            
            # Execute research iteration
            try:
                result = crew.kickoff()
                print(f"✅ Iteration {research_state.current_search_count} completed")
                
                # Check if we should continue
                if research_state.current_search_count >= max_searches:
                    break
                    
            except Exception as e:
                print(f"❌ Error in research iteration: {str(e)}")
                break
        
        # Generate final report
        print("\n📝 Generating final report...")
        final_crew = self._create_final_report_crew()
        final_result = final_crew.kickoff()
        
        return research_state.final_summary
    
    def _create_research_crew(self) -> Crew:
        """Create crew for research iteration"""
        # Determine if this is first search or follow-up
        if research_state.current_search_count == 0:
            # First search - planning and initial search
            tasks = [
                create_research_planning_task(research_state.topic, research_state.max_web_searches),
                create_web_search_task(),
                create_analysis_task()
            ]
        else:
            # Follow-up search - generate new query, search, analyze
            tasks = [
                Task(
                    description=f"""
                    Generate a follow-up search query to fill knowledge gaps in the current research.
                    
                    Current research summary: {research_state.summaries[-1] if research_state.summaries else 'None'}
                    
                    Generate a targeted query that will help gather additional information not covered in existing research.
                    """,
                    expected_output="Follow-up search query generated",
                    agent=create_research_coordinator()
                ),
                create_web_search_task(),
                create_analysis_task()
            ]
        
        return Crew(
            agents=[create_research_coordinator(), create_web_researcher(), create_content_analyst()],
            tasks=tasks,
            process=Process.sequential,
            verbose=True
        )
    
    def _create_final_report_crew(self) -> Crew:
        """Create crew for final report generation"""
        return Crew(
            agents=[create_content_analyst()],
            tasks=[create_final_report_task()],
            process=Process.sequential,
            verbose=True
        )

# Interactive Demo
def main():
    """Main function for interactive demo"""
    print("\n" + "=" * 60)
    print("🔬 Research Assistant with CrewAI")
    print("=" * 60)
    
    print("\nThis demo showcases a research assistant that can:")
    print("1. Generate targeted search queries")
    print("2. Perform comprehensive web searches")
    print("3. Analyze and summarize findings")
    print("4. Create detailed research reports")
    print("5. Use CrewAI's multi-agent system for workflow orchestration")
    
    # Example research topics
    example_topics = [
        "How was DeepSeek R1 trained",
        "Latest developments in quantum computing",
        "Impact of AI on healthcare industry",
        "Sustainable energy solutions 2024"
    ]
    
    print("\n📋 Example research topics:")
    for i, topic in enumerate(example_topics, 1):
        print(f"{i}. {topic}")
    
    print("\nStarting research with example topic...")
    
    # Initialize research assistant
    assistant = ResearchAssistantCrew()
    
    # Conduct research on example topic
    topic = example_topics[0]
    max_searches = 3
    
    try:
        final_report = assistant.conduct_research(topic, max_searches)
        
        print("\n" + "=" * 60)
        print("📊 RESEARCH COMPLETE")
        print("=" * 60)
        print(final_report)
        
    except Exception as e:
        print(f"\n❌ Error during research: {str(e)}")
        print("Please check your API keys and try again.")

if __name__ == "__main__":
    main()