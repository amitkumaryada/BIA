from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
import os
from dotenv import load_dotenv
import asyncio

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

async def main():
    # Initialize the model client (e.g., using OpenAI GPT-4 model).
    model_client = OpenAIChatCompletionClient(model="gpt-4o")  # api_key can be set via env or passed here.

    # Create an assistant agent using the model (no external tools in this basic example).
    assistant = AssistantAgent(
        name="assistant",
        model_client=model_client,
        system_message="You are a helpful AI assistant."
    )

    try:
        # Have the agent respond to a simple user prompt.
        result = await assistant.run(task="Hello, how are you today?")
        print("Assistant's response:")
        print(result.messages[-1].content)  # Print the assistant's reply.
    finally:
        # Close the model client session.
        await model_client.close()

# Run the async function
if __name__ == "__main__":
    asyncio.run(main())
