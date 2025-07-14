import datetime
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv
import asyncio
import os

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")


# Define two tool functions for the agent (they will be auto-wrapped as FunctionTool).
async def add_numbers(x: float, y: float) -> float:
    """Add two numbers and return the sum."""
    return x + y

async def get_current_time() -> str:
    """Get the current date and time as a string."""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


async def main():
    # Initialize the model client and create an agent with both tools.
    model_client = OpenAIChatCompletionClient(model="gpt-3.5-turbo")
    multi_tool_agent = AssistantAgent(
        name="multi_tool_agent",
        model_client=model_client,
        tools=[add_numbers, get_current_time],  # Provide multiple tools in the list
        system_message="You are a math and time assistant. Use tools if needed to answer accurately.",
        reflect_on_tool_use=True  # Instruct agent to summarize raw tool outputs into natural language
    )

    try:
        # Example usage: ask a question that may require a tool.
        result = await multi_tool_agent.run(task="What is 123 + 456? and what the time now")
        print(result.messages[-1].content)  # The agent should use the add_numbers tool to compute the sum.
    finally:
        # Ensure the model client is closed properly
        await model_client.close()

# Run the async function
if __name__ == "__main__":
    asyncio.run(main())
