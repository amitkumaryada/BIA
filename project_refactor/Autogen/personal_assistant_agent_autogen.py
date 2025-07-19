"""    Task:
    Create a Personal Assistant Agent using AutoGen that can:
    Have basic conversations
    Remember user's name and preferences across sessions
    Perform simple calculations
    Tell current time/date
"""

import os
import getpass
import datetime
import math
import json
import asyncio
from typing import Dict, List, Any, Callable
from dotenv import load_dotenv

# Import AutoGen components using correct imports
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.tools import FunctionTool


def setup_environment():
    """Setup your API KEYs"""
    # Load environment variables from .env file
    load_dotenv()
    
    # Check for API key and prompt if not available
    if not os.environ.get("OPENAI_API_KEY"):
        openai_key = getpass.getpass("Enter your OPENAI API KEY")
        os.environ["OPENAI_API_KEY"] = openai_key

    print("Environment setup completed")


setup_environment()

# Create a class to store user preferences and memory
class UserMemory:
    def __init__(self):
        self.memory = {}
    
    def save(self, key, value):
        self.memory[key] = value
    
    def get(self, key, default=None):
        return self.memory.get(key, default)
    
    def clear(self):
        self.memory = {}

    def get_all(self):
        return self.memory


# Initialize user memory
user_memory = UserMemory()


# Helper functions for calculations and time
def get_current_time() -> str:
    """Get the current time and date."""
    return f"Current time is {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"


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


# Create the memory management function
def update_memory(message: str):
    """Update user memory based on message content"""
    # If name is mentioned, store it
    if "my name is" in message.lower():
        try:
            name = message.lower().split("my name is")[1].strip().split()[0].capitalize()
            user_memory.save("name", name)
            print(f"Memory updated: Name = {name}")
        except Exception as e:
            print(f"Error extracting name: {e}")
            pass
        
    # Look for preferences
    if "i like" in message.lower():
        try:
            preference = message.lower().split("i like")[1].strip()
            user_memory.save("preference", preference)
            print(f"Memory updated: Preference = {preference}")
        except Exception as e:
            print(f"Error extracting preference: {e}")
            pass


# Create the system message with memory awareness
def get_system_message():
    """Create a system message that includes any stored memory about the user"""
    base_message = """You are a helpful personal assistant that can have conversations, 
    remember user information, perform calculations, and provide the current time. 
    Always try to be friendly and helpful.
    
    When asked about time, respond with the current time.
    When asked to calculate something, solve the math expression carefully.
    
    Use a conversational and friendly tone."""
    
    memory_context = ""
    if user_memory.get("name"):
        memory_context += f"\nThe user's name is {user_memory.get('name')}. "
    if user_memory.get("preference"):
        memory_context += f"The user likes {user_memory.get('preference')}. "
    
    return base_message + memory_context


# Custom input function that updates memory
def custom_input_func(prompt=""):
    user_input = input(prompt)
    update_memory(user_input)
    
    # Check if user is asking for time or calculations
    if "time" in user_input.lower() or "what time" in user_input.lower() or "current time" in user_input.lower():
        print(f"[Tool] {get_current_time()}")
    
    if "calculate" in user_input.lower():
        try:
            expression = user_input.lower().split("calculate")[1].strip()
            result = calculator(expression)
            print(f"[Tool] {result}")
        except Exception as e:
            print(f"[Tool] Error processing calculation: {e}")
    
    return user_input


# Define a message handler that processes incoming messages
class MessageHandler:
    def __init__(self, memory_manager):
        self.memory_manager = memory_manager
        
    async def process_message(self, message, sender_name):
        # Process memory from user messages
        if sender_name == "user_proxy" and isinstance(message, str):
            self.memory_manager(message)
            
        # Return the original message unmodified
        return message


async def main():
    # Initialize model client for the assistant
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please make sure OPENAI_API_KEY is set in the environment.")
        
    model_client = OpenAIChatCompletionClient(model="gpt-4o", temperature=0.2)
    
    # Create message handler that manages memory
    message_handler = MessageHandler(update_memory)
    
    try:
        # Create an assistant agent with our custom system message
        assistant = AssistantAgent(
            name="personal_assistant",
            model_client=model_client, 
            system_message=get_system_message(),
        )
        
        # Create user proxy that will use our custom input function
        user_proxy = UserProxyAgent(
            name="user_proxy", 
            input_func=custom_input_func
        )
        
        # Define termination condition - stop when user types "GOODBYE"
        termination_condition = TextMentionTermination("GOODBYE")
        
        # Create a team with the assistant and user
        team = RoundRobinGroupChat(
            [assistant, user_proxy],
            termination_condition=termination_condition
        )
        
        # Process initial message to extract user info
        initial_message = "Hello! my name is amit I take lectures on BIA"
        update_memory(initial_message)
        
        print("Personal Assistant Agent (AutoGen) is ready!")
        print("Type 'GOODBYE' to end the conversation\n")
        
        # Run the team conversation and stream to console
        stream = team.run_stream(task=initial_message)
        await Console(stream)
        
    finally:
        # Ensure the model client is closed properly
        await model_client.close()


# Run the conversation when script is executed
if __name__ == "__main__":
    asyncio.run(main())
