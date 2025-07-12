from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv
import asyncio
import os

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

async def main():
    # Initialize model client for the assistant.
    model_client = OpenAIChatCompletionClient(model="gpt-3.5-turbo")

    try:
        # Create an assistant agent and a user proxy agent for live feedback.
        assistant = AssistantAgent(name="assistant", model_client=model_client, system_message="You are a helpful assistant.")
        user_proxy = UserProxyAgent(name="user_proxy", input_func=input)  # Uses input() to capture human feedback from console.

        # Define a termination condition: stop when the user (via proxy) types "APPROVE".
        termination_condition = TextMentionTermination("APPROVE")

        # Create a team with the assistant and the user proxy agent.
        team = RoundRobinGroupChat([assistant, user_proxy], termination_condition=termination_condition)

        # Run the team conversation and stream messages to the console.
        stream = team.run_stream(task="Write a 4-line poem about the ocean.")
        await Console(stream)
    finally:
        # Ensure the model client is closed properly
        await model_client.close()

# Run the async function
if __name__ == "__main__":
    asyncio.run(main())
