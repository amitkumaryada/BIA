"""
### Customer Service Bot with Human Escalation (AutoGen Version)

An AutoGen-based customer service bot that replicates the LangGraph functionality:
1. Classify customer queries (common vs complex)
2. Handle common queries automatically
3. Create support tickets for complex issues
4. Escalate to human agents when needed
5. Manage ticket creation and tracking
"""

import os
import asyncio
import uuid
import json
import datetime
from typing import Annotated, Dict, List, Any, Optional
from enum import Enum
from dotenv import load_dotenv

# Import AutoGen components
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient

# Load environment variables
load_dotenv()

# Define ticket status enum
class TicketStatus(str, Enum):
    NEW = "new"
    IN_PROGRESS = "in_progress"
    WAITING_FOR_HUMAN = "waiting_for_human"
    WAITING_FOR_CUSTOMER = "waiting_for_customer"
    RESOLVED = "resolved"
    CLOSED = "closed"

# Global state for customer service session
customer_service_state = {
    "current_ticket": None,
    "issue_resolved": False,
    "needs_human": False,
    "customer_id": None,
    "conversation_history": [],
    "tickets_db": {}  # Simple in-memory ticket storage
}

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

# Tool functions for customer service
def classify_query(query: Annotated[str, "Customer query to classify"]) -> str:
    """
    Classify if a customer query is common (can be handled automatically) or complex (needs human intervention).
    
    Args:
        query: The customer query to classify
        
    Returns:
        JSON string with classification results
    """
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

def handle_common_query(query: Annotated[str, "Common query to handle"]) -> str:
    """
    Handle common customer service queries automatically.
    
    Args:
        query: The customer query to handle
        
    Returns:
        Response to the common query
    """
    query_lower = query.lower()
    
    # Find the best matching response
    for category, response in COMMON_QUERIES_KB.items():
        if any(keyword in query_lower for keyword in category.split()):
            customer_service_state["issue_resolved"] = True
            return f"{response}\n\nIs there anything else I can help you with today?"
    
    # If no exact match, provide a general helpful response
    return "I'd be happy to help you with that. Let me connect you with additional resources or create a support ticket for personalized assistance."

def create_support_ticket(query: Annotated[str, "Customer query that needs a support ticket"]) -> str:
    """
    Create a support ticket for complex issues that need human assistance.
    
    Args:
        query: The customer query that needs a ticket
        
    Returns:
        Ticket creation confirmation with details
    """
    # Generate ticket details
    ticket_id = str(uuid.uuid4())[:8].upper()
    customer_id = customer_service_state.get("customer_id", "GUEST_" + str(uuid.uuid4())[:6])
    
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
        "customer_id": customer_id,
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
    customer_service_state["tickets_db"][ticket_id] = ticket
    customer_service_state["current_ticket"] = ticket
    customer_service_state["needs_human"] = True
    
    return json.dumps({
        "ticket_created": True,
        "ticket_id": ticket_id,
        "priority": priority,
        "category": category,
        "status": "created",
        "message": f"I've created support ticket #{ticket_id} for your inquiry. A human agent will review this and get back to you soon."
    }, indent=2)

def get_ticket_status(ticket_id: Annotated[str, "Ticket ID to check status for"]) -> str:
    """
    Get the status of a support ticket.
    
    Args:
        ticket_id: The ticket ID to check
        
    Returns:
        Ticket status information
    """
    ticket = customer_service_state["tickets_db"].get(ticket_id)
    
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

def escalate_to_human(reason: Annotated[str, "Reason for escalation"]) -> str:
    """
    Escalate the current conversation to a human agent.
    
    Args:
        reason: Reason for escalation
        
    Returns:
        Escalation confirmation message
    """
    customer_service_state["needs_human"] = True
    
    escalation_message = (
        f"I'm escalating your inquiry to a human agent because: {reason}\n\n"
        "A human support specialist will take over this conversation shortly. "
        "They will have access to your conversation history and any tickets created."
    )
    
    if customer_service_state["current_ticket"]:
        ticket_id = customer_service_state["current_ticket"]["ticket_id"]
        escalation_message += f"\n\nYour support ticket #{ticket_id} has been updated with this escalation."
    
    return escalation_message

def get_conversation_summary() -> str:
    """
    Get a summary of the current customer service session.
    
    Returns:
        Summary of the conversation and any tickets
    """
    summary = {
        "session_active": True,
        "issue_resolved": customer_service_state["issue_resolved"],
        "needs_human_intervention": customer_service_state["needs_human"],
        "current_ticket": customer_service_state["current_ticket"],
        "total_tickets_created": len(customer_service_state["tickets_db"])
    }
    
    return json.dumps(summary, indent=2)

async def main():
    """Main function to run the customer service bot."""
    
    # Check for required API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please set the OPENAI_API_KEY environment variable.")
    
    # Initialize customer ID
    customer_service_state["customer_id"] = "CUSTOMER_" + str(uuid.uuid4())[:8]
    
    # Initialize model client
    model_client = OpenAIChatCompletionClient(model="gpt-4o", temperature=0.1)
    
    try:
        # Create the customer service assistant
        assistant = AssistantAgent(
            name="customer_service_agent",
            model_client=model_client,
            system_message=(
                "You are a helpful customer service assistant. Your goal is to provide excellent customer support.\n\n"
                "Your workflow:\n"
                "1. Use classify_query to determine if a query is common or complex\n"
                "2. For common queries, use handle_common_query to provide immediate assistance\n"
                "3. For complex issues, use create_support_ticket to create a ticket\n"
                "4. Use escalate_to_human if the customer is frustrated or needs specialized help\n"
                "5. Use get_ticket_status to check on existing tickets\n"
                "6. Use get_conversation_summary to review the session status\n\n"
                "Guidelines:\n"
                "- Always be polite, professional, and empathetic\n"
                "- Try to resolve issues quickly when possible\n"
                "- Create tickets for complex issues that need human review\n"
                "- Escalate to humans when customers are frustrated or need specialized help\n"
                "- Keep responses concise but helpful\n"
                "- When you've resolved an issue or completed a task, say 'TASK COMPLETE'"
            ),
            tools=[classify_query, handle_common_query, create_support_ticket, 
                   get_ticket_status, escalate_to_human, get_conversation_summary]
        )
        
        # Create user proxy that handles human input
        user_proxy = UserProxyAgent(
            name="customer",
            input_func=input  # This will prompt for user input
        )
        
        # Define termination condition
        termination_condition = TextMentionTermination("TASK COMPLETE")
        
        # Create the team
        team = RoundRobinGroupChat(
            [assistant, user_proxy],
            termination_condition=termination_condition
        )
        
        # Start the customer service session
        print("\n🎧 Customer Service Bot (AutoGen Version)")
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
        
        # Run the conversation with streaming
        initial_task = "Hello! I'm your customer service assistant. How can I help you today?"
        stream = team.run_stream(task=initial_task)
        await Console(stream)
        
        # Display session summary
        print("\n" + "=" * 50)
        print("📊 SESSION SUMMARY")
        print("=" * 50)
        
        summary = json.loads(get_conversation_summary())
        print(f"Issue Resolved: {'✅' if summary['issue_resolved'] else '❌'}")
        print(f"Human Intervention Needed: {'⚠️' if summary['needs_human_intervention'] else '✅'}")
        print(f"Tickets Created: {summary['total_tickets_created']}")
        
        if customer_service_state["current_ticket"]:
            ticket = customer_service_state["current_ticket"]
            print(f"\n🎫 Active Ticket: #{ticket['ticket_id']}")
            print(f"   Priority: {ticket['priority']}/5")
            print(f"   Category: {ticket['category'].title()}")
            print(f"   Status: {ticket['status'].title()}")
            
    finally:
        # Ensure the model client is closed properly
        await model_client.close()

if __name__ == "__main__":
    asyncio.run(main())
