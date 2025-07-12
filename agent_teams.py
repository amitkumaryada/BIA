from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
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
    # Initialize a shared model client for both agents (using the same LLM).
    model_client = OpenAIChatCompletionClient(model="gpt-3.5-turbo")

    # Create two AssistantAgents with distinct roles.
    primary_agent = AssistantAgent(
        name="primary",
        model_client=model_client,
        system_message="You are a creative writer who produces a draft response."
    )
    critic_agent = AssistantAgent(
        name="critic",
        model_client=model_client,
        system_message="You are a critical reviewer. Provide feedback and say 'APPROVE' when the draft is good."
    )

    # Define a termination condition to stop when the critic agent says "APPROVE".
    termination = TextMentionTermination("APPROVE")

    # Assemble the team with round-robin turn-taking.
    team = RoundRobinGroupChat([primary_agent, critic_agent], termination_condition=termination)

    # Run the team on a task. Each agent will take turns contributing to the conversation.
    result = await team.run(task="Write a short poem about the fall season.")
    for msg in result.messages:
        print(f"{msg.source}: {getattr(msg, 'content', '')}")
    await model_client.close()

# Run the async function
if __name__ == "__main__":
    asyncio.run(main())
    