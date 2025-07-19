"""
### Personal Assistant Agent (CrewAI Version)

A CrewAI-based personal assistant that can:
1. Have basic conversations
2. Remember user's name and preferences across sessions
3. Perform simple calculations
4. Tell current time/date
5. Provide helpful assistance with various tasks
"""

import os
import getpass
import datetime
import math
import json
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field

# Setup environment variables
def setup_environment():
    """Setup your API KEYs"""
    if not os.environ.get("OPENAI_API_KEY"):
        openai_key = getpass.getpass("Enter your OPENAI API KEY: ")
        os.environ["OPENAI_API_KEY"] = openai_key

    if not os.environ.get("LANGSMITH_API_KEY"):
        langsmith_key = getpass.getpass("ENTER your LangSmith API KEY (optional): ")
        if langsmith_key:
            os.environ["LANGSMITH_API_KEY"] = langsmith_key
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_PROJECT"] = "Personal Assistant CrewAI"
        else:
            print("Skipping LangSmith setup")

    print("Environment setup completed")

setup_environment()

# Import CrewAI components
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain_core.callbacks import BaseCallbackHandler

# Define data models
class UserProfile(BaseModel):
    name: Optional[str] = None
    preferences: Dict[str, Any] = Field(default_factory=dict)
    conversation_history: List[Dict[str, str]] = Field(default_factory=list)
    session_count: int = 0

class AssistantState:
    """Global state for personal assistant session"""
    def __init__(self):
        self.user_profile = UserProfile()
        self.current_conversation: List[Dict[str, str]] = []
        self.session_active = True
        self.last_interaction = datetime.datetime.now()

# Global state instance
assistant_state = AssistantState()

# CrewAI Tools
class TimeAndDateTool(BaseTool):
    name: str = "get_current_time"
    description: str = "Get the current time and date"
    
    def _run(self) -> str:
        """Get the current time and date"""
        current_time = datetime.datetime.now()
        return f"Current time is {current_time.strftime('%Y-%m-%d %H:%M:%S')} ({current_time.strftime('%A, %B %d, %Y')})"

class CalculatorTool(BaseTool):
    name: str = "calculator"
    description: str = "Evaluate mathematical expressions safely"
    
    def _run(self, expression: str) -> str:
        """Evaluate a mathematical expression safely"""
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

class MemoryTool(BaseTool):
    name: str = "remember_user_info"
    description: str = "Remember user information and preferences"
    
    def _run(self, info_type: str, info_value: str) -> str:
        """Remember user information"""
        if info_type.lower() == "name":
            assistant_state.user_profile.name = info_value
            return f"I'll remember that your name is {info_value}!"
        else:
            assistant_state.user_profile.preferences[info_type] = info_value
            return f"I've noted that your {info_type} is {info_value}."

class ConversationHistoryTool(BaseTool):
    name: str = "get_conversation_context"
    description: str = "Get context from previous conversations"
    
    def _run(self) -> str:
        """Get conversation context"""
        context = []
        
        if assistant_state.user_profile.name:
            context.append(f"User's name: {assistant_state.user_profile.name}")
        
        if assistant_state.user_profile.preferences:
            context.append("User preferences:")
            for key, value in assistant_state.user_profile.preferences.items():
                context.append(f"  - {key}: {value}")
        
        if assistant_state.user_profile.conversation_history:
            context.append(f"Previous conversations: {len(assistant_state.user_profile.conversation_history)} sessions")
        
        return "\n".join(context) if context else "No previous context available."

class UserPreferencesTool(BaseTool):
    name: str = "get_user_preferences"
    description: str = "Get stored user preferences and information"
    
    def _run(self) -> str:
        """Get user preferences"""
        if not assistant_state.user_profile.name and not assistant_state.user_profile.preferences:
            return "I don't have any stored information about you yet. Feel free to tell me about yourself!"
        
        info = []
        if assistant_state.user_profile.name:
            info.append(f"Name: {assistant_state.user_profile.name}")
        
        if assistant_state.user_profile.preferences:
            info.append("Preferences:")
            for key, value in assistant_state.user_profile.preferences.items():
                info.append(f"  - {key}: {value}")
        
        return "\n".join(info)

# Initialize tools
time_tool = TimeAndDateTool()
calculator_tool = CalculatorTool()
memory_tool = MemoryTool()
context_tool = ConversationHistoryTool()
preferences_tool = UserPreferencesTool()

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0.1)

# Define CrewAI Agents
personal_assistant_agent = Agent(
    role="Personal Assistant",
    goal="Provide helpful, friendly, and personalized assistance to users while remembering their preferences and maintaining conversation context",
    backstory="You are a knowledgeable and friendly personal assistant who excels at helping users with various tasks, remembering their preferences, and maintaining engaging conversations. You can perform calculations, provide time information, and remember user details across sessions.",
    tools=[time_tool, calculator_tool, memory_tool, context_tool, preferences_tool],
    llm=llm,
    verbose=True,
    memory=True
)

conversation_manager_agent = Agent(
    role="Conversation Manager",
    goal="Manage conversation flow and ensure context is maintained across interactions",
    backstory="You are responsible for managing conversation flow, ensuring user context is preserved, and coordinating with the personal assistant to provide seamless interactions.",
    tools=[context_tool, preferences_tool],
    llm=llm,
    verbose=True,
    memory=True
)

# Define Tasks
def create_assistance_task(user_input: str) -> Task:
    return Task(
        description=f"Help the user with their request: '{user_input}'. Use available tools as needed to provide accurate information, perform calculations, or remember user preferences. Be friendly and personalized in your response.",
        expected_output="A helpful and personalized response to the user's request",
        agent=personal_assistant_agent,
        tools=[time_tool, calculator_tool, memory_tool, context_tool, preferences_tool]
    )

def create_context_management_task(user_input: str) -> Task:
    return Task(
        description=f"Analyze the user input '{user_input}' and determine if any user information should be remembered or if previous context should be retrieved to enhance the response.",
        expected_output="Context analysis and any relevant user information that should be remembered",
        agent=conversation_manager_agent,
        tools=[context_tool, preferences_tool]
    )

# Main Personal Assistant Crew
class PersonalAssistantCrew:
    def __init__(self):
        self.conversation_count = 0
        
    def process_user_input(self, user_input: str) -> str:
        """Process user input using the CrewAI workflow"""
        try:
            # Increment conversation count
            self.conversation_count += 1
            assistant_state.last_interaction = datetime.datetime.now()
            
            # Add to conversation history
            assistant_state.current_conversation.append({
                "role": "user",
                "content": user_input,
                "timestamp": datetime.datetime.now().isoformat()
            })
            
            # Extract user information if present
            self._extract_user_info(user_input)
            
            # Create tasks
            assistance_task = create_assistance_task(user_input)
            
            # Create crew and execute
            crew = Crew(
                agents=[personal_assistant_agent],
                tasks=[assistance_task],
                process=Process.sequential,
                verbose=True,
                memory=True
            )
            
            result = crew.kickoff()
            response = str(result)
            
            # Add to conversation history
            assistant_state.current_conversation.append({
                "role": "assistant",
                "content": response,
                "timestamp": datetime.datetime.now().isoformat()
            })
            
            return response
            
        except Exception as e:
            return f"I apologize, but I encountered an error while processing your request: {str(e)}. Please try again."
    
    def _extract_user_info(self, user_input: str):
        """Extract and store user information from input"""
        input_lower = user_input.lower()
        
        # Extract name
        if "my name is" in input_lower or "i'm" in input_lower or "i am" in input_lower:
            words = user_input.split()
            for i, word in enumerate(words):
                if word.lower() in ["is", "i'm", "am"] and i + 1 < len(words):
                    potential_name = words[i + 1].strip(".,!?")
                    if potential_name.isalpha():
                        assistant_state.user_profile.name = potential_name.title()
                        break
        
        # Extract preferences (simple pattern matching)
        preference_patterns = [
            ("i like", "likes"),
            ("i love", "loves"),
            ("i prefer", "prefers"),
            ("i work", "work"),
            ("i teach", "teaches"),
            ("i study", "studies")
        ]
        
        for pattern, key in preference_patterns:
            if pattern in input_lower:
                start_idx = input_lower.find(pattern) + len(pattern)
                preference = user_input[start_idx:].strip()
                if preference:
                    assistant_state.user_profile.preferences[key] = preference
                break

# Interactive Personal Assistant Function
def interactive_personal_assistant():
    """Run an interactive personal assistant session"""
    print("\n🤖 Personal Assistant (CrewAI Version)")
    print("=" * 50)
    print("Hello! I'm your personal assistant. I can help you with:")
    print("• Basic conversations and questions")
    print("• Mathematical calculations")
    print("• Current time and date")
    print("• Remembering your preferences")
    print("• And much more!")
    print("\nType 'quit' to end our conversation.\n")
    
    # Example interactions
    print("💡 Try saying:")
    print("• 'Hello, my name is [Your Name]'")
    print("• 'What time is it?'")
    print("• 'Calculate 15 * 23 + 7'")
    print("• 'I like programming and AI'")
    print("• 'What do you remember about me?'\n")
    
    crew = PersonalAssistantCrew()
    
    # Load previous session if exists
    if assistant_state.user_profile.name:
        print(f"Welcome back, {assistant_state.user_profile.name}! 👋")
    
    while True:
        try:
            # Get user input
            user_input = input("\n👤 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye']:
                farewell = f"Goodbye{', ' + assistant_state.user_profile.name if assistant_state.user_profile.name else ''}! It was great chatting with you. Have a wonderful day! 👋"
                print(f"\n🤖 Assistant: {farewell}")
                break
            
            if not user_input:
                print("I'm here to help! What would you like to know or do?")
                continue
            
            # Process the input
            print("\n🤖 Processing...")
            response = crew.process_user_input(user_input)
            
            # Display response
            print(f"\n🤖 Assistant: {response}")
            
        except KeyboardInterrupt:
            print("\n\n👋 Session ended. Thank you for using the personal assistant!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Let's try again with a new question.")
    
    # Save conversation to history
    if assistant_state.current_conversation:
        assistant_state.user_profile.conversation_history.append({
            "session_id": len(assistant_state.user_profile.conversation_history) + 1,
            "timestamp": datetime.datetime.now().isoformat(),
            "messages": assistant_state.current_conversation.copy()
        })
        assistant_state.user_profile.session_count += 1
    
    # Display session summary
    print("\n" + "=" * 50)
    print("📊 SESSION SUMMARY")
    print("=" * 50)
    print(f"Session Duration: {crew.conversation_count} interactions")
    print(f"User Name: {assistant_state.user_profile.name or 'Not provided'}")
    print(f"Stored Preferences: {len(assistant_state.user_profile.preferences)}")
    print(f"Total Sessions: {assistant_state.user_profile.session_count}")
    
    if assistant_state.user_profile.preferences:
        print("\n📝 Your Preferences:")
        for key, value in assistant_state.user_profile.preferences.items():
            print(f"   • {key}: {value}")

# Example usage and testing
if __name__ == "__main__":
    # Example interactions to demonstrate functionality
    examples = [
        "Hello, my name is Amit and I teach BIA courses",
        "What time is it?",
        "Calculate 25 * 4 + 100",
        "I love working with AI and machine learning",
        "What do you remember about me?"
    ]
    
    print("\n=== Personal Assistant with Memory (CrewAI) ===")
    print("\nThis demo showcases a personal assistant that can:")
    print("1. Have natural conversations")
    print("2. Remember user information across sessions")
    print("3. Perform mathematical calculations")
    print("4. Provide current time and date")
    print("5. Store and recall user preferences")
    print("6. Use CrewAI's agent system for intelligent responses\n")
    
    print("Example interactions you can try:")
    for i, example in enumerate(examples, 1):
        print(f"{i}. {example}")
    
    print("\nStarting interactive conversation...")
    
    # Start the interactive personal assistant
    interactive_personal_assistant()
