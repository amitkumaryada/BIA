from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
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
    # Start a conversation with an assistant agent.
    model_client1 = OpenAIChatCompletionClient(model="gpt-3.5-turbo")
    assistant1 = AssistantAgent(name="assistant", system_message="You are a helpful assistant.", model_client=model_client1)

    # Agent generates a response to the initial user prompt.
    initial_result = await assistant1.run(task="Write a 3-line poem about the sunrise.")
    print("Initial answer:\n", initial_result.messages[-1].content)
    # Save the agent's internal state (conversation context, etc.).
    agent_state = await assistant1.save_state()  # Serialize the conversation state:contentReference[oaicite:28]{index=28}
    await model_client1.close()

    # ... (Later or in a new session) Recreate the agent and load the saved state.
    model_client2 = OpenAIChatCompletionClient(model="gpt-3.5-turbo")
    assistant2 = AssistantAgent(name="assistant", system_message="You are a helpful assistant.", model_client=model_client2)
    await assistant2.load_state(agent_state)     # Restore the conversation state into the new agent:contentReference[oaicite:29]{index=29}

    # Continue the conversation knowing what was said before.
    followup_result = await assistant2.run(task="What was the last line of the poem you just wrote?")
    print("Follow-up answer:\n", followup_result.messages[-1].content)
    await model_client2.close()

# Run the async function
if __name__ == "__main__":
    asyncio.run(main()) 