"""
### Customer Service Bot with Human Escalation (CrewAI Version)

A CrewAI-based customer service bot that replicates the LangGraph functionality:
1. Classify customer queries (common vs complex)
2. Handle common queries automatically
3. Create support tickets for complex issues
4. Escalate to human agents when needed
5. Manage ticket creation and tracking
"""

import os
import getpass
import uuid
import json
import datetime
from typing import Dict, List, Any, Optional
from enum import Enum
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
            os.environ["LANGCHAIN_PROJECT"] = "Customer Service Bot CrewAI"
        else:
            print("Skipping LangSmith setup")

    print("Environment setup completed")

setup_environment()

# Import CrewAI components
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain_core.callbacks import BaseCallbackHandler

# Define ticket status enum
class TicketStatus(str, Enum):
    NEW = "new"
    IN_PROGRESS = "in_progress"
    WAITING_FOR_HUMAN = "waiting_for_human"
    WAITING_FOR_CUSTOMER = "waiting_for_customer"
    RESOLVED = "resolved"
    CLOSED = "closed"

# Define data models
class QueryClassification(BaseModel):
    is_common_query: bool = Field(description="Whether this is a common query that can be handled automatically")
    confidence_score: float = Field(description="Confidence score between 0 and 1")
    category: str = Field(description="Category of the customer query: billing, technical, account, product, other")
    estimated_complexity: int = Field(description="Estimated complexity from 1-5, where 5 is most complex")

class Ticket(BaseModel):
    ticket_id: str
    customer_id: str
    subject: str
    description: str
    status: TicketStatus
    priority: int
    created_at: str
    updated_at: str
    assigned_to: Optional[str] = None
    category: Optional[str] = None

class CustomerServiceState:
    """Global state for customer service session"""
    def __init__(self):
        self.current_ticket: Optional[Dict] = None
        self.issue_resolved: bool = False
        self.needs_human: bool = False
        self.customer_id: str = f"CUSTOMER_{uuid.uuid4().hex[:8].upper()}"
        self.conversation_history: List[Dict] = []
        self.tickets_db: Dict[str, Dict] = {}  # Simple in-memory ticket storage
        self.response: str = ""

# Global state instance
customer_state = CustomerServiceState()

# Mock knowledge base for common queries
COMMON_QUERIES_KB = {
    "business hours": "Our business hours are Monday-Friday 9 AM to 6 PM EST, and Saturday 10 AM to 4 PM EST. We're closed on Sundays.",
    "contact": "You can reach us at support@company.com or call 1-800-SUPPORT (1-800-786-7678).",
    "shipping": "We offer standard shipping (5-7 business days) for $5.99, expedited shipping (2-3 business days) for $12.99, and next-day delivery for $24.99.",
    "return policy": "We accept returns within 30 days of purchase. Items must be in original condition with tags attached. Return shipping is free for defective items.",
    "payment methods": "We accept all major credit cards (Visa, MasterCard, American Express, Discover), PayPal, Apple Pay, and Google Pay.",
    "account": "You can manage your account by logging into our website and clicking 'My Account' in the top right corner.",
    "password reset": "To reset your password, go to the login page and click 'Forgot Password'. Enter your email address and we'll send you a reset link.",
    "order status": "You can check your order status by logging into your account and viewing 'Order History', or by using our order tracking tool with your order number."
}

# CrewAI Tools
class QueryClassificationTool(BaseTool):
    name: str = "classify_query"
    description: str = "Classify if a customer query is common (can be handled automatically) or complex (needs human intervention)"
    
    def _run(self, query: str) -> str:
        """Classify the customer query"""
        query_lower = query.lower()
        
        # Check if it's a common query
        is_common = False
        matched_category = "other"
        confidence = 0.0
        
        for category, response in COMMON_QUERIES_KB.items():
            if any(keyword in query_lower for keyword in category.split()):
                is_common = True
                matched_category = category
                confidence = 0.9
                break
        
        # Determine complexity based on query characteristics
        complexity_indicators = [
            "frustrated", "angry", "refund", "charged twice", "doesn't work", 
            "broken", "defective", "urgent", "immediately", "complaint",
            "cancel", "dispute", "error", "problem", "issue", "help"
        ]
        
        complexity_score = sum(1 for indicator in complexity_indicators if indicator in query_lower)
        estimated_complexity = min(5, max(1, complexity_score + 1))
        
        # If complexity is high, it's not a common query
        if estimated_complexity >= 3:
            is_common = False
            confidence = 0.8
        
        classification = {
            "is_common_query": is_common,
            "confidence_score": confidence,
            "category": matched_category,
            "estimated_complexity": estimated_complexity,
            "query": query
        }
        
        return json.dumps(classification, indent=2)

class CommonQueryHandlerTool(BaseTool):
    name: str = "handle_common_query"
    description: str = "Handle common customer service queries automatically"
    
    def _run(self, query: str) -> str:
        """Handle common customer queries"""
        query_lower = query.lower()
        
        # Find the best matching response
        for category, response in COMMON_QUERIES_KB.items():
            if any(keyword in query_lower for keyword in category.split()):
                customer_state.issue_resolved = True
                customer_state.response = f"{response}\n\nIs there anything else I can help you with today?"
                return customer_state.response
        
        # If no exact match, provide a general helpful response
        return "I'd be happy to help you with that. Let me connect you with additional resources or create a support ticket for personalized assistance."

class TicketCreationTool(BaseTool):
    name: str = "create_support_ticket"
    description: str = "Create a support ticket for complex issues that need human assistance"
    
    def _run(self, query: str) -> str:
        """Create a support ticket"""
        # Generate ticket details
        ticket_id = str(uuid.uuid4())[:8].upper()
        
        # Determine priority based on query content
        high_priority_keywords = ["urgent", "immediately", "asap", "emergency", "critical", "angry", "frustrated"]
        priority = 4 if any(keyword in query.lower() for keyword in high_priority_keywords) else 3
        
        # Categorize the query
        if any(word in query.lower() for word in ["bill", "charge", "payment", "refund"]):
            category = "billing"
        elif any(word in query.lower() for word in ["password", "login", "account", "access"]):
            category = "account"
        elif any(word in query.lower() for word in ["broken", "error", "doesn't work", "technical"]):
            category = "technical"
        elif any(word in query.lower() for word in ["product", "item", "order", "delivery"]):
            category = "product"
        else:
            category = "other"
        
        # Create ticket
        ticket = {
            "ticket_id": ticket_id,
            "customer_id": customer_state.customer_id,
            "subject": f"Customer inquiry - {category.title()}",
            "description": query,
            "status": TicketStatus.NEW.value,
            "priority": priority,
            "category": category,
            "created_at": datetime.datetime.now().isoformat(),
            "updated_at": datetime.datetime.now().isoformat(),
            "assigned_to": None
        }
        
        # Store ticket
        customer_state.tickets_db[ticket_id] = ticket
        customer_state.current_ticket = ticket
        customer_state.needs_human = True
        
        return json.dumps({
            "ticket_created": True,
            "ticket_id": ticket_id,
            "priority": priority,
            "category": category,
            "status": "created",
            "message": f"I've created support ticket #{ticket_id} for your inquiry. A human agent will review this and get back to you soon."
        }, indent=2)

class HumanEscalationTool(BaseTool):
    name: str = "escalate_to_human"
    description: str = "Escalate the current conversation to a human agent"
    
    def _run(self, reason: str) -> str:
        """Escalate to human agent"""
        customer_state.needs_human = True
        
        escalation_message = (
            f"I'm escalating your inquiry to a human agent because: {reason}\n\n"
            "A human support specialist will take over this conversation shortly. "
            "They will have access to your conversation history and any tickets created."
        )
        
        if customer_state.current_ticket:
            ticket_id = customer_state.current_ticket["ticket_id"]
            escalation_message += f"\n\nYour support ticket #{ticket_id} has been updated with this escalation."
        
        return escalation_message

class TicketStatusTool(BaseTool):
    name: str = "get_ticket_status"
    description: str = "Get the status of a support ticket"
    
    def _run(self, ticket_id: str) -> str:
        """Get ticket status"""
        ticket = customer_state.tickets_db.get(ticket_id)
        
        if not ticket:
            return f"Ticket #{ticket_id} not found. Please check the ticket ID and try again."
        
        return json.dumps({
            "ticket_id": ticket["ticket_id"],
            "status": ticket["status"],
            "priority": ticket["priority"],
            "category": ticket["category"],
            "created_at": ticket["created_at"],
            "subject": ticket["subject"]
        }, indent=2)

# Initialize tools
classify_tool = QueryClassificationTool()
common_query_tool = CommonQueryHandlerTool()
ticket_tool = TicketCreationTool()
escalation_tool = HumanEscalationTool()
status_tool = TicketStatusTool()

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0.1)

# Define CrewAI Agents
query_classifier_agent = Agent(
    role="Query Classifier",
    goal="Classify customer queries to determine if they can be handled automatically or need human intervention",
    backstory="You are an expert at analyzing customer service queries and determining their complexity and urgency.",
    tools=[classify_tool],
    llm=llm,
    verbose=True
)

customer_service_agent = Agent(
    role="Customer Service Representative",
    goal="Provide excellent customer service by handling common queries and creating tickets for complex issues",
    backstory="You are a friendly and professional customer service representative with extensive knowledge of company policies and procedures.",
    tools=[common_query_tool, ticket_tool, escalation_tool, status_tool],
    llm=llm,
    verbose=True
)

human_escalation_agent = Agent(
    role="Human Escalation Specialist",
    goal="Handle complex customer issues that require human intervention and specialized knowledge",
    backstory="You are a senior customer service specialist who handles the most complex and sensitive customer issues.",
    tools=[escalation_tool, status_tool],
    llm=llm,
    verbose=True
)

# Define Tasks
def create_classification_task(customer_query: str) -> Task:
    return Task(
        description=f"Classify the following customer query to determine if it's a common query that can be handled automatically or if it needs human intervention: '{customer_query}'",
        expected_output="A JSON classification with is_common_query, confidence_score, category, and estimated_complexity",
        agent=query_classifier_agent,
        tools=[classify_tool]
    )

def create_service_task(customer_query: str, classification: str) -> Task:
    return Task(
        description=f"Based on the classification '{classification}', handle the customer query: '{customer_query}'. If it's a common query, provide an immediate response. If it's complex, create a support ticket.",
        expected_output="A helpful response to the customer, either resolving their issue or confirming ticket creation",
        agent=customer_service_agent,
        tools=[common_query_tool, ticket_tool, escalation_tool, status_tool]
    )

def create_escalation_task(issue_details: str) -> Task:
    return Task(
        description=f"Handle the escalated customer issue: '{issue_details}'. Provide specialized assistance and update ticket status as needed.",
        expected_output="A comprehensive response addressing the complex customer issue",
        agent=human_escalation_agent,
        tools=[escalation_tool, status_tool]
    )

# Custom Callback Handler for Human-in-the-Loop
class HumanInterventionCallback(BaseCallbackHandler):
    def __init__(self):
        self.human_input_required = False
        self.human_response = ""
    
    def on_agent_action(self, action, **kwargs):
        # Check if human intervention is needed
        if customer_state.needs_human and not self.human_input_required:
            self.human_input_required = True
            print("\n🚨 HUMAN INTERVENTION REQUIRED 🚨")
            print("This issue requires specialized human assistance.")
            
            if customer_state.current_ticket:
                ticket = customer_state.current_ticket
                print(f"\n📋 Ticket Details:")
                print(f"   ID: #{ticket['ticket_id']}")
                print(f"   Priority: {ticket['priority']}/5")
                print(f"   Category: {ticket['category'].title()}")
                print(f"   Status: {ticket['status'].title()}")
            
            # Get human input
            self.human_response = input("\n👤 Support Agent Response: ")
            customer_state.needs_human = False
            
            # Update ticket status if exists
            if customer_state.current_ticket:
                customer_state.current_ticket["status"] = TicketStatus.IN_PROGRESS.value
                customer_state.current_ticket["updated_at"] = datetime.datetime.now().isoformat()

# Main Customer Service Crew
class CustomerServiceCrew:
    def __init__(self):
        self.callback_handler = HumanInterventionCallback()
        
    def handle_customer_query(self, customer_query: str) -> str:
        """Handle a customer query using the CrewAI workflow"""
        try:
            # Reset state for new query
            customer_state.issue_resolved = False
            customer_state.needs_human = False
            customer_state.response = ""
            
            # Step 1: Classify the query
            classification_task = create_classification_task(customer_query)
            classification_crew = Crew(
                agents=[query_classifier_agent],
                tasks=[classification_task],
                process=Process.sequential,
                verbose=True
            )
            
            classification_result = classification_crew.kickoff()
            
            # Step 2: Handle the query based on classification
            service_task = create_service_task(customer_query, str(classification_result))
            service_crew = Crew(
                agents=[customer_service_agent],
                tasks=[service_task],
                process=Process.sequential,
                verbose=True
            )
            
            service_result = service_crew.kickoff()
            
            # Step 3: Check if human escalation is needed
            if customer_state.needs_human and not self.callback_handler.human_input_required:
                self.callback_handler.on_agent_action(None)
                
                # Create escalation task with human input
                escalation_task = create_escalation_task(f"{customer_query}\n\nHuman Agent Response: {self.callback_handler.human_response}")
                escalation_crew = Crew(
                    agents=[human_escalation_agent],
                    tasks=[escalation_task],
                    process=Process.sequential,
                    verbose=True
                )
                
                escalation_result = escalation_crew.kickoff()
                return str(escalation_result)
            
            return str(service_result)
            
        except Exception as e:
            return f"I apologize, but I encountered an error while processing your request: {str(e)}. Please try again or contact support directly."

# Interactive Customer Service Function
def interactive_customer_service():
    """Run an interactive customer service session"""
    print("\n🎧 Customer Service Bot (CrewAI Version)")
    print("=" * 50)
    print("Welcome! I'm here to help you with your questions and concerns.")
    print("I can handle common queries instantly or create support tickets for complex issues.")
    print("Type 'quit' to end the conversation.\n")
    
    # Example queries for demonstration
    print("💡 Try asking about:")
    print("• Business hours")
    print("• Shipping information")
    print("• Return policy")
    print("• Account issues")
    print("• Technical problems\n")
    
    crew = CustomerServiceCrew()
    
    while True:
        try:
            # Get customer input
            customer_input = input("\n👤 You: ").strip()
            
            if customer_input.lower() in ['quit', 'exit', 'bye']:
                print("\n👋 Thank you for contacting customer service. Have a great day!")
                break
            
            if not customer_input:
                print("Please enter your question or concern.")
                continue
            
            # Add to conversation history
            customer_state.conversation_history.append({
                "role": "customer",
                "content": customer_input,
                "timestamp": datetime.datetime.now().isoformat()
            })
            
            # Process the query
            print("\n🤖 Processing your request...")
            response = crew.handle_customer_query(customer_input)
            
            # Display response
            print(f"\n🎧 Bot: {response}")
            
            # Add to conversation history
            customer_state.conversation_history.append({
                "role": "assistant",
                "content": response,
                "timestamp": datetime.datetime.now().isoformat()
            })
            
            # Display ticket information if created
            if customer_state.current_ticket:
                ticket = customer_state.current_ticket
                print(f"\n📋 Ticket #{ticket['ticket_id']} | Priority: {ticket['priority']}/5 | Status: {ticket['status'].title()}")
            
            # Display session summary
            if customer_state.issue_resolved:
                print("\n✅ Issue resolved! Is there anything else I can help you with?")
                
        except KeyboardInterrupt:
            print("\n\n👋 Session ended. Thank you for using our customer service!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Let's try again with a new question.")
    
    # Display final session summary
    print("\n" + "=" * 50)
    print("📊 SESSION SUMMARY")
    print("=" * 50)
    print(f"Customer ID: {customer_state.customer_id}")
    print(f"Issue Resolved: {'✅' if customer_state.issue_resolved else '❌'}")
    print(f"Human Intervention Used: {'⚠️' if customer_state.needs_human else '✅'}")
    print(f"Tickets Created: {len(customer_state.tickets_db)}")
    print(f"Conversation Messages: {len(customer_state.conversation_history)}")
    
    if customer_state.tickets_db:
        print("\n🎫 Created Tickets:")
        for ticket_id, ticket in customer_state.tickets_db.items():
            print(f"   #{ticket_id} - {ticket['category'].title()} (Priority: {ticket['priority']}/5)")

# Example usage
if __name__ == "__main__":
    # Example customer queries to demonstrate functionality
    examples = [
        # Common query that can be handled automatically
        "What are your business hours?",
        
        # Technical issue that needs human intervention
        "I've been trying to reset my password for 3 days and the reset email never arrives. I've checked my spam folder multiple times and tried with different email addresses. This is very frustrating!",
        
        # Billing dispute requiring human intervention
        "I was charged twice for my monthly subscription. I need this refunded immediately.",
        
        # Simple product question
        "Do you offer next-day delivery?"
    ]
    
    print("\n=== Customer Service Bot with Human-in-the-Loop (CrewAI) ===")
    print("\nThis demo showcases a customer service bot that can:")
    print("1. Handle common queries automatically")
    print("2. Create support tickets for complex issues")
    print("3. Escalate to human agents when needed")
    print("4. Use CrewAI's multi-agent system for comprehensive customer service\n")
    
    print("Example queries you can try:")
    for i, example in enumerate(examples, 1):
        print(f"{i}. {example}")
    
    print("\nStarting interactive conversation...")
    
    # Start the interactive customer service bot
    interactive_customer_service()
