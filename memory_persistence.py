from autogen_agentchat.agents import AssistantAgent
from autogen_core.memory import ListMemory, MemoryContent, MemoryMimeType
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv
import asyncio
import os

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

# Define a tool that provides weather info, respecting a units parameter.
async def get_weather(city: str, units: str = "imperial") -> str:
    """Return weather for the city, in specified units."""
    if units == "imperial":
        return f"The weather in {city} is 73°F and Sunny."
    elif units == "metric":
        return f"The weather in {city} is 23°C and Sunny."
    else:
        return "Sorry, I don't have weather info for those units."

async def main():
    # Initialize a memory store for user preferences.
    user_memory = ListMemory()
    # Add a preference to the memory (e.g., use metric units for weather).
    await user_memory.add(MemoryContent(content="The weather should be in metric units", 
                                        mime_type=MemoryMimeType.TEXT))
    
    # Create an assistant agent that uses the memory and the weather tool.
    model_client = OpenAIChatCompletionClient(model="gpt-3.5-turbo")
    assistant_agent = AssistantAgent(
        name="assistant_agent",
        model_client=model_client,
        tools=[get_weather],
        memory=[user_memory],  # Attach the memory store to the agent
        system_message="You are a weather assistant. Provide weather info using the user's preferences."
    )

    try:
        # Ask a question; the agent will retrieve relevant info from memory before responding.
        result = await assistant_agent.run(task="What is the weather in New York?")
        print(result.messages[-1].content)  # Expected to use metric units (°C) due to the stored preference.
    finally:
        # Ensure the model client is closed properly
        await model_client.close()

# Run the async function
if __name__ == "__main__":
    asyncio.run(main())
