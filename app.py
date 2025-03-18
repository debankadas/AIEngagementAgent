import os
from typing import Dict, List, Optional, Any, Tuple, TypedDict
import json
from datetime import datetime
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# LangChain and LangGraph imports
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.memory import ConversationBufferMemory
from langchain.tools import tool
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.callbacks.manager import CallbackManagerForToolRun, AsyncCallbackManagerForToolRun

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode


# For OCR functionality
import pytesseract
from PIL import Image
import io
import base64

# For email and calendar integration (would be actual implementations in production)
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

load_dotenv()

# State schema
class LeadState(TypedDict):
    conversation_history: List[Dict]
    lead_info: Dict
    current_stage: str
    business_context: Dict
    session_id: str
    error: Optional[str]


# Data models
class LeadProfile(BaseModel):
    full_name: str = Field(description="Full name of the lead")
    company: str = Field(description="Company name")
    role: str = Field(description="Job title/role")
    email: Optional[str] = Field(None, description="Email address")
    phone: Optional[str] = Field(None, description="Phone number")
    location: Optional[str] = Field(None, description="Location/city")
    website: Optional[str] = Field(None, description="Website URL")
    products_of_interest: List[str] = Field(default_factory=list, description="Products the lead is interested in")
    moq: Optional[str] = Field(None, description="Minimum order quantity")
    concerns: List[str] = Field(default_factory=list, description="Pain points or concerns")
    preferred_contact: Optional[str] = Field(None, description="Preferred contact method")
    interest_level: Optional[str] = Field(None, description="Interest level (Low, Medium, High)")
    follow_up_actions: List[str] = Field(default_factory=list, description="Required follow-up actions")


# System prompts
SYSTEM_PROMPTS = {
    "greeting": """
You are Jane, an AI assistant for {company_name} at a trade show. The exhibitor team is currently busy with other visitors.
Your goal is to engage with visitors, collect their information, and understand their needs.

Current conversation stage: GREETING
In this stage, you should:
1. Introduce yourself as an AI assistant for {company_name}
2. Ask what brings the visitor to the trade show/booth
3. Be friendly, professional, and concise

Company information:
{company_info}
""",

    "info_gathering": """
You are Jane, an AI assistant for {company_name} at a trade show.
Your goal is to gather information about the visitor.

Current conversation stage: INFORMATION GATHERING
In this stage, you should:
1. Ask about their company, role, and location
2. Understand their specific needs and interests related to our products
3. Ask if they'd be willing to show their business card for quick information capture
4. Be friendly, professional, and concise

What we know about the lead so far:
{lead_info_summary}

Company information:
{company_info}
""",

    "business_card": """
You are Jane, an AI assistant for {company_name} at a trade show.
Your goal is to scan the visitor's business card and verify the information.

Current conversation stage: BUSINESS CARD SCANNING
In this stage, you should:
1. Thank them for sharing their business card
2. Confirm the scanned information is correct
3. Ask about their preferred contact method for follow-ups (email, WhatsApp, etc.)
4. Be friendly, professional, and concise

What we know about the lead so far:
{lead_info_summary}

Company information:
{company_info}
""",

    "product_discussion": """
You are Jane, an AI assistant for {company_name} at a trade show.
Your goal is to discuss specific products that match the visitor's needs.

Current conversation stage: PRODUCT DISCUSSION
In this stage, you should:
1. Present relevant products based on their stated interests
2. Ask about minimum order quantities (MOQ) they're looking for
3. Inquire about their timeline or urgency
4. Probe for any specific requirements they might have
5. Be friendly, professional, and concise

What we know about the lead so far:
{lead_info_summary}

Products and services information:
{products_info}

Company information:
{company_info}
""",

    "objection_handling": """
You are Jane, an AI assistant for {company_name} at a trade show.
Your goal is to address any concerns or objections the visitor might have.

Current conversation stage: OBJECTION HANDLING
In this stage, you should:
1. Listen for any concerns about pricing, quality, delivery, etc.
2. Address objections with relevant information
3. Offer alternatives if available
4. Be empathetic, helpful, and solution-oriented
5. Be friendly, professional, and concise

What we know about the lead so far:
{lead_info_summary}

Products and services information:
{products_info}

Company information:
{company_info}
""",

    "next_steps": """
You are Jane, an AI assistant for {company_name} at a trade show.
Your goal is to establish clear next steps with the visitor.

Current conversation stage: NEXT STEPS
In this stage, you should:
1. Suggest appropriate follow-up actions (samples, demo, meeting, pricing info)
2. Confirm their preferred communication method
3. Ask if they'd be interested in joining our newsletter or receiving updates
4. Be friendly, professional, and concise

What we know about the lead so far:
{lead_info_summary}

Products and services information:
{products_info}

Company information:
{company_info}
""",

    "closing": """
You are Jane, an AI assistant for {company_name} at a trade show.
Your goal is to professionally close the conversation.

Current conversation stage: CLOSING
In this stage, you should:
1. Thank them for visiting the booth
2. Summarize what was discussed and the next steps
3. Confirm when they can expect to hear from the team
4. Wish them a successful rest of the trade show
5. Be friendly, professional, and concise

What we know about the lead so far:
{lead_info_summary}

Next steps agreed:
{follow_up_summary}

Company information:
{company_info}
"""
}

CONVERSATION_ANALYZER_PROMPT = """
Analyze the conversation between the AI assistant and the trade show visitor.
Extract and categorize key information into a structured lead profile.

Focus on extracting:
1. Basic information (name, company, role, contact details)
2. Products or services of interest
3. Minimum order quantity (MOQ) if mentioned
4. Pain points, concerns, or objections
5. Interest level (Low, Medium, High) based on engagement
6. Follow-up preferences and specific requests
7. Any special requirements or notes

Conversation transcript:
{conversation}

Format your response as a valid JSON object matching the LeadProfile model.
"""


# Tools for the agent
class BusinessCardTool:
    """Tool for scanning business cards and extracting contact information"""

    @tool("scan_business_card")
    def scan_business_card(image_data: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """
        Scan a business card image and extract contact information.

        Args:
            image_data: Base64 encoded string of the business card image

        Returns:
            JSON string with extracted contact information
        """
        # In a production environment, this would use OCR to scan the card
        # For this example, we'll simulate the extraction

        # Simulated OCR result
        contact_info = {
            "full_name": "Rohan Singh",
            "company": "Café Brew Haven",
            "role": "Owner",
            "email": "rohan@brewhavengroup.com",
            "phone": "+91 98765 43210",
            "website": "www.brewhavengroup.com",
            "location": "Pune, Maharashtra, India"
        }

        return json.dumps(contact_info)


class DatabaseTool:
    """Tool for interacting with the lead database"""

    @tool("save_lead_to_database")
    def save_lead_to_database(lead_data: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """
        Save lead information to the database.

        Args:
            lead_data: JSON string with lead information

        Returns:
            Confirmation message
        """
        # In production, this would connect to Firebase/PostgreSQL
        # For this example, we'll simulate the database operation

        try:
            # Parse the lead data
            lead_info = json.loads(lead_data)

            # In production: db.collection('leads').add(lead_info)

            return f"Successfully saved lead information for {lead_info.get('full_name', 'the visitor')} to the database."
        except Exception as e:
            return f"Error saving lead to database: {str(e)}"


class SchedulingTool:
    """Tool for scheduling follow-up actions"""

    @tool("schedule_follow_up")
    def schedule_follow_up(details: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """
        Schedule a follow-up action such as a meeting or email.

        Args:
            details: JSON string with follow-up details including type (email/meeting),
                    contact information, and scheduling preferences

        Returns:
            Confirmation message
        """
        # In production, this would integrate with Google Calendar API
        # For this example, we'll simulate the scheduling

        try:
            # Parse the follow-up details
            follow_up = json.loads(details)

            follow_up_type = follow_up.get("type", "email")
            contact_name = follow_up.get("contact_name", "the visitor")

            if follow_up_type == "meeting":
                # In production: google_calendar.create_event(...)
                return f"Meeting with {contact_name} scheduled for {follow_up.get('datetime', 'the requested time')}."
            elif follow_up_type == "email":
                # In production: email_service.schedule_email(...)
                return f"Follow-up email to {contact_name} scheduled to be sent {follow_up.get('send_time', 'as requested')}."
            else:
                return f"Unknown follow-up type: {follow_up_type}"
        except Exception as e:
            return f"Error scheduling follow-up: {str(e)}"


class EmailTool:
    """Tool for sending emails to leads"""

    @tool("send_email")
    def send_email(email_details: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """
        Send an email to a lead.

        Args:
            email_details: JSON string with email details including recipient,
                          subject, body, and any attachments

        Returns:
            Confirmation message
        """
        # In production, this would integrate with an email service
        # For this example, we'll simulate sending an email

        try:
            # Parse the email details
            details = json.loads(email_details)

            recipient = details.get("recipient", "")
            subject = details.get("subject", "")

            # In production: email_service.send_email(...)

            return f"Email with subject '{subject}' sent to {recipient}."
        except Exception as e:
            return f"Error sending email: {str(e)}"


class TranscriptionTool:
    """Tool for transcribing and analyzing conversations"""

    @tool("analyze_conversation")
    def analyze_conversation(conversation: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """
        Analyze a conversation to extract structured lead information.

        Args:
            conversation: JSON string with conversation history

        Returns:
            Structured lead profile as JSON
        """
        # later we will update this to use an LLM to analyze the conversation


        try:
            # Parse the conversation
            conv_data = json.loads(conversation)

            # Format the conversation for analysis
            formatted_conversation = "\n".join([
                f"{msg['role'].capitalize()}: {msg['content']}"
                for msg in conv_data
            ])

            # In production, this would be a call to an LLM
            # For this example, we'll return a simulated result
            lead_profile = {
                "full_name": "Rohan Singh",
                "company": "Café Brew Haven",
                "role": "Owner",
                "email": "rohan@brewhavengroup.com",
                "phone": "+91 98765 43210",
                "location": "Pune, India",
                "products_of_interest": ["Vanilla Blend", "Cold Brew Blend", "Premium Single-Origin"],
                "moq": "50kg for trial, 5kg sample packs first",
                "concerns": ["Price sensitivity", "Product quality consistency"],
                "preferred_contact": "WhatsApp",
                "interest_level": "Medium-High",
                "follow_up_actions": [
                    "Send price comparison with current supplier",
                    "Provide sample packs of Vanilla and Cold Brew blends",
                    "Schedule follow-up call in 2 weeks"
                ]
            }

            return json.dumps(lead_profile)
        except Exception as e:
            return f"Error analyzing conversation: {str(e)}"


# Main component functions
def initialize_state(session_id: str, company_info: Dict) -> LeadState:
    """
    Initialize the agent state with company information
    """
    return {
        "conversation_history": [],
        "lead_info": {},
        "transcription": [],
        "follow_up_tasks": [],
        "current_stage": "greeting",
        "business_context": company_info,
        "session_id": session_id,
        "error": None
    }


def get_llm(model_name="gpt-4o-mini"):
    """Get the LLM for the agent"""
    return ChatOpenAI(model=model_name, temperature=0)


def format_lead_info_summary(lead_info: Dict) -> str:
    """Format lead info for inclusion in prompts"""
    if not lead_info:
        return "No information collected yet."

    # Build a formatted summary of what we know
    summary_parts = []

    if "full_name" in lead_info:
        summary_parts.append(f"Name: {lead_info['full_name']}")

    if "company" in lead_info:
        summary_parts.append(f"Company: {lead_info['company']}")

    if "role" in lead_info:
        summary_parts.append(f"Role: {lead_info['role']}")

    if "email" in lead_info:
        summary_parts.append(f"Email: {lead_info['email']}")

    if "phone" in lead_info:
        summary_parts.append(f"Phone: {lead_info['phone']}")

    if "location" in lead_info:
        summary_parts.append(f"Location: {lead_info['location']}")

    if "products_of_interest" in lead_info and lead_info["products_of_interest"]:
        products = ", ".join(lead_info["products_of_interest"])
        summary_parts.append(f"Products of Interest: {products}")

    if "moq" in lead_info and lead_info["moq"]:
        summary_parts.append(f"MOQ: {lead_info['moq']}")

    if "concerns" in lead_info and lead_info["concerns"]:
        concerns = ", ".join(lead_info["concerns"])
        summary_parts.append(f"Concerns: {concerns}")

    if "preferred_contact" in lead_info and lead_info["preferred_contact"]:
        summary_parts.append(f"Preferred Contact Method: {lead_info['preferred_contact']}")

    if not summary_parts:
        return "Basic information collection in progress."

    return "\n".join(summary_parts)


def format_follow_up_summary(follow_up_tasks: List[Dict]) -> str:
    """Format follow-up tasks for inclusion in prompts"""
    if not follow_up_tasks:
        return "No follow-up tasks defined yet."

    # Extract relevant follow-up information
    summaries = []

    for task in follow_up_tasks:
        task_type = task.get("type", "unknown")

        if task_type == "email":
            summaries.append(f"Follow-up email to be sent: {task.get('details', 'No details')}")
        elif task_type == "meeting":
            summaries.append(f"Meeting to be scheduled: {task.get('details', 'No details')}")
        elif task_type == "sample":
            summaries.append(f"Product samples to be sent: {task.get('details', 'No details')}")
        elif task_type == "call":
            summaries.append(f"Follow-up call: {task.get('details', 'No details')}")
        else:
            summaries.append(f"{task_type.capitalize()}: {task.get('details', 'No details')}")

    return "\n".join(summaries)


def create_tools():
    """Create the tools the agent can use"""
    business_card_tool = BusinessCardTool()
    database_tool = DatabaseTool()
    scheduling_tool = SchedulingTool()
    email_tool = EmailTool()
    transcription_tool = TranscriptionTool()

    return [
        business_card_tool.scan_business_card,
        database_tool.save_lead_to_database,
        scheduling_tool.schedule_follow_up,
        email_tool.send_email,
        transcription_tool.analyze_conversation
    ]


def handle_user_input(state: LeadState, user_input: str) -> LeadState:
    """
    Process user input and add it to conversation history
    """
    # Add user input to conversation history
    state["conversation_history"].append({
        "role": "human",
        "content": user_input,
        "timestamp": datetime.now().isoformat()
    })

    return state





def update_conversation_stage(state: LeadState) -> LeadState:
    """
    Update the conversation stage based on the current state
    """
    current_stage = state["current_stage"]
    conversation_length = len(state["conversation_history"])

    # Stage progression logic
    # In a real system, this would use NLU to understand context and detect stage transitions
    stage_transitions = {
        "greeting": "info_gathering",
        "info_gathering": "business_card",
        "business_card": "product_discussion",
        "product_discussion": "objection_handling",
        "objection_handling": "next_steps",
        "next_steps": "closing",
        "closing": "completed"
    }

    # Simple heuristic for stage progression
    # In a production, we need to make this more sophisticated
    if current_stage in stage_transitions:
        # Check for stage-specific conditions
        if current_stage == "greeting" and conversation_length >= 4:
            state["current_stage"] = stage_transitions[current_stage]
        elif current_stage == "info_gathering" and "full_name" in state["lead_info"]:
            state["current_stage"] = stage_transitions[current_stage]
        elif current_stage == "business_card" and "email" in state["lead_info"]:
            state["current_stage"] = stage_transitions[current_stage]
        elif current_stage == "product_discussion" and "products_of_interest" in state["lead_info"]:
            state["current_stage"] = stage_transitions[current_stage]
        elif current_stage == "objection_handling" and "concerns" in state["lead_info"]:
            state["current_stage"] = stage_transitions[current_stage]
        elif current_stage == "next_steps" and state["follow_up_tasks"]:
            state["current_stage"] = stage_transitions[current_stage]
        elif current_stage == "closing" and conversation_length >= 14:  # Arbitrary threshold
            state["current_stage"] = stage_transitions[current_stage]

    return state


def generate_conversation_summary(state: LeadState) -> LeadState:
    """
    Generate a summary of the conversation for the database
    """
    # In a real system, this would use an LLM to analyze the full conversation
    # For this example, we'll use our existing lead_info

    if not state["lead_info"]:
        # If we don't have lead info, analyze the conversation
        transcription_tool = TranscriptionTool()
        conversation_json = json.dumps(state["conversation_history"])
        lead_profile_json = transcription_tool.analyze_conversation(conversation_json)

        try:
            lead_profile = json.loads(lead_profile_json)
            state["lead_info"] = lead_profile
        except json.JSONDecodeError:
            # Handle parsing error
            state["error"] = "Failed to parse lead profile from conversation analysis"

    # Add a timestamp to the lead info
    state["lead_info"]["conversation_timestamp"] = datetime.now().isoformat()
    state["lead_info"]["event_name"] = state["business_context"].get("event_name", "Trade Show")

    return state


def create_follow_up_email(state: LeadState) -> LeadState:
    """
    Create a follow-up email based on the conversation
    """
    if not state["lead_info"]:
        return state

    # Get an LLM for generating the email
    llm = get_llm()

    # Create a prompt for the email
    email_prompt = """
You are creating a follow-up email for a lead from a trade show. 
Write a personalized email that references their specific interests and concerns.

Lead Information:
Name: {name}
Company: {company}
Role: {role}
Products of Interest: {products}
Concerns/Objections: {concerns}

The email should:
1. Thank them for visiting your booth at {event_name}
2. Reference specific products they were interested in
3. Address any concerns they mentioned
4. Propose clear next steps
5. Be professional but warm in tone
6. Include a specific call to action
7. Keep it concise (150-200 words)

Write the full email with subject line and body.
    """

    # Format the necessary information
    lead_name = state["lead_info"].get("full_name", "Valued Customer")
    company = state["lead_info"].get("company", "your company")
    role = state["lead_info"].get("role", "your role")
    products = ", ".join(state["lead_info"].get("products_of_interest", ["our products"]))
    concerns = ", ".join(state["lead_info"].get("concerns", ["No specific concerns mentioned"]))
    event_name = state["business_context"].get("event_name", "the trade show")

    # Generate the email
    email_content = llm.invoke(
        email_prompt.format(
            name=lead_name,
            company=company,
            role=role,
            products=products,
            concerns=concerns,
            event_name=event_name
        )
    ).content

    # Add the email to follow-up tasks
    state["follow_up_tasks"].append({
        "type": "email",
        "status": "draft",
        "recipient": state["lead_info"].get("email", ""),
        "recipient_name": lead_name,
        "content": email_content,
        "created_at": datetime.now().isoformat()
    })

    return state


def schedule_follow_up_meeting(state: LeadState) -> LeadState:
    """
    Schedule a follow-up meeting based on the conversation
    """
    if not state["lead_info"] or "follow_up_actions" not in state["lead_info"]:
        return state

    # Check if any follow-up actions mention a meeting or call
    meeting_needed = False
    meeting_details = ""

    for action in state["lead_info"]["follow_up_actions"]:
        if "meeting" in action.lower() or "call" in action.lower():
            meeting_needed = True
            meeting_details = action
            break

    if meeting_needed:
        # Add a meeting task
        state["follow_up_tasks"].append({
            "type": "meeting",
            "status": "pending",
            "contact_name": state["lead_info"].get("full_name", "Lead"),
            "contact_email": state["lead_info"].get("email", ""),
            "details": meeting_details,
            "created_at": datetime.now().isoformat()
        })

    return state


def save_to_database(state: LeadState) -> LeadState:
    """
    Save the lead information to the database
    """
    # In a real system, this would connect to a database
    # For this example, we'll simulate the database operation

    # Create a database-ready record
    lead_record = {
        "basic_info": {
            "full_name": state["lead_info"].get("full_name", ""),
            "company": state["lead_info"].get("company", ""),
            "role": state["lead_info"].get("role", ""),
            "email": state["lead_info"].get("email", ""),
            "phone": state["lead_info"].get("phone", ""),
            "location": state["lead_info"].get("location", ""),
            "website": state["lead_info"].get("website", "")
        },
        "business_details": {
            "products_of_interest": state["lead_info"].get("products_of_interest", []),
            "moq": state["lead_info"].get("moq", ""),
            "concerns": state["lead_info"].get("concerns", []),
            "interest_level": state["lead_info"].get("interest_level", "")
        },
        "engagement": {
            "event_name": state["business_context"].get("event_name", "Trade Show"),
            "conversation_timestamp": state["lead_info"].get("conversation_timestamp", ""),
            "follow_up_actions": state["lead_info"].get("follow_up_actions", []),
            "preferred_contact": state["lead_info"].get("preferred_contact", "")
        },
        "follow_up_tasks": state["follow_up_tasks"],
        "conversation_history": state["conversation_history"]
    }

    # In a real system:
    # db.collection('leads').add(lead_record)

    # For this example, we'll just acknowledge the save
    state["lead_info"]["saved_to_database"] = True

    return state


def process_input(state, message):
    """Process user input and update state"""
    # Add user input to conversation history
    if "conversation_history" not in state:
        state["conversation_history"] = []

    state["conversation_history"].append({
        "role": "human",
        "content": message,
        "timestamp": datetime.now().isoformat()
    })

    return state


def generate_response(state):
    """Generate a more sophisticated response based on conversation stage"""
    # Get the current stage
    current_stage = state.get("current_stage", "greeting")

    # Simple stage-based responses
    if current_stage == "greeting":
        response = generate_response_with_llm(state)
        # Move to next stage
        state["current_stage"] = "info_gathering"
    elif current_stage == "info_gathering":
        response = generate_response_with_llm(state)
        # Move to next stage
        state["current_stage"] = "product_discussion"
    elif current_stage == "product_discussion":
        response = generate_response_with_llm(state)
        # Move to next stage
        state["current_stage"] = "closing"
    else:
        response = "Thank you for your interest! Someone from our team will follow up with you soon."

    # Add response to conversation history
    state["conversation_history"].append({
        "role": "assistant",
        "content": response,
        "timestamp": datetime.now().isoformat()
    })

    return state


def generate_response_with_llm(state):
    """Generate a response using LangChain and an LLM"""
    # Get conversation context
    current_stage = state.get("current_stage", "greeting")
    company_info = state.get("business_context", {}).get("company_info", "")

    # Create system prompt based on stage
    if current_stage == "greeting":
        system_prompt = f"You are an AI assistant at a trade show booth for {company_info}. Greet the visitor warmly and ask what brings them to the show."
    elif current_stage == "info_gathering":
        system_prompt = "Ask the visitor about their company, role, and what products they're interested in."
    else:
        system_prompt = "Be helpful and professional in your response."

    # Create the prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", state["conversation_history"][-1]["content"])
    ])


    # Set up the LLM
    llm = ChatOpenAI(model="gpt-4o-mini")

    # Generate the response
    chain = prompt | llm
    result = chain.invoke({"input": state["conversation_history"][-1]["content"]})

    # Add response to conversation history
    state["conversation_history"].append({
        "role": "assistant",
        "content": result.content,
        "timestamp": datetime.now().isoformat()
    })

    # Advance the stage (simplified)
    if current_stage == "greeting":
        state["current_stage"] = "info_gathering"
    elif current_stage == "info_gathering":
        state["current_stage"] = "product_discussion"
    elif current_stage == "product_discussion":
        state["current_stage"] = "closing"

    return state

# Build the LangGraph
def build_trade_show_agent_graph():
    """
    Build a simplified LangGraph without checkpointing
    """
    # Create the graph
    builder = StateGraph(LeadState)

    # Add nodes
    builder.add_node("handle_user_input", lambda state, message: process_input(state, message))
    builder.add_node("generate_response", generate_response)

    # Add edges
    builder.add_edge("handle_user_input", "generate_response")
    builder.add_edge("generate_response", "handle_user_input")

    # Set the entry point
    builder.set_entry_point("handle_user_input")

    # Compile without checkpointing
    return builder.compile()


# API interface for the application
def create_api():
    """
    Create a FastAPI application for the trade show agent
    This would be expanded in a real implementation
    """
    from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel

    app = FastAPI(title="Trade Show Lead Management Agent")

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # In production, specify exact origins
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Models for API requests
    class InitSessionRequest(BaseModel):
        company_name: str
        event_name: str
        company_info: str
        products_info: str

    class MessageRequest(BaseModel):
        session_id: str
        message: str

    # Session storage
    active_sessions = {}

    @app.post("/api/init-session")
    async def init_session(request: InitSessionRequest):
        """Initialize a new conversation session"""
        # Generate a session ID
        session_id = f"session_{datetime.now().strftime('%Y%m%d%H%M%S')}_{len(active_sessions) + 1}"

        # Create the business context
        business_context = {
            "company_name": request.company_name,
            "event_name": request.event_name,
            "company_info": request.company_info,
            "products_info": request.products_info
        }

        # Initialize the state
        state = initialize_state(session_id, business_context)

        # Create the graph
        graph = build_trade_show_agent_graph()

        # Store the session
        active_sessions[session_id] = {
            "graph": graph,
            "state": state
        }

        return {"session_id": session_id, "status": "initialized"}

    @app.post("/api/send-message")
    async def send_message(request: MessageRequest):
        """Send a message to the agent"""
        session_id = request.session_id

        # Check if the session exists
        if session_id not in active_sessions:
            raise HTTPException(status_code=404, detail="Session not found")

        # Get the session
        session = active_sessions[session_id]
        graph = session["graph"]
        state = session["state"]

        # Process the message
        state = handle_user_input(state, request.message)

        # Run the graph
        new_state = graph.invoke(state)

        # Update the session state
        active_sessions[session_id]["state"] = new_state

        # Get the assistant's response
        assistant_messages = [
            msg for msg in new_state["conversation_history"]
            if msg["role"] == "assistant"
        ]

        last_message = assistant_messages[-1] if assistant_messages else {"content": "No response generated"}

        return {
            "response": last_message["content"],
            "current_stage": new_state["current_stage"]
        }

    @app.websocket("/ws/{session_id}")
    async def websocket_endpoint(websocket: WebSocket, session_id: str):
        """WebSocket endpoint for real-time conversation"""
        await websocket.accept()

        # Check if the session exists
        if session_id not in active_sessions:
            await websocket.send_json({"error": "Session not found"})
            await websocket.close()
            return

        # Get the session
        session = active_sessions[session_id]
        graph = session["graph"]
        state = session["state"]

        try:
            while True:
                # Receive message from client
                data = await websocket.receive_text()

                # Process the message
                state = handle_user_input(state, data)

                # Run the graph
                new_state = graph.invoke(state)

                # Update the session state
                active_sessions[session_id]["state"] = new_state

                # Get the assistant's response
                assistant_messages = [
                    msg for msg in new_state["conversation_history"]
                    if msg["role"] == "assistant"
                ]

                last_message = assistant_messages[-1] if assistant_messages else {"content": "No response generated"}

                # Send response to client
                await websocket.send_json({
                    "response": last_message["content"],
                    "current_stage": new_state["current_stage"]
                })

                # Check if the conversation is completed
                if new_state["current_stage"] == "completed":
                    await websocket.send_json({"status": "conversation_completed"})
                    break

        except WebSocketDisconnect:
            # Handle disconnect
            pass

    @app.get("/api/export-lead/{session_id}")
    async def export_lead(session_id: str):
        """Export lead data for a session"""
        # Check if the session exists
        if session_id not in active_sessions:
            raise HTTPException(status_code=404, detail="Session not found")

        # Get the session
        session = active_sessions[session_id]
        state = session["state"]

        # Generate the lead summary if not already done
        if state["current_stage"] != "completed":
            state = generate_conversation_summary(state)
            state = create_follow_up_email(state)
            state = schedule_follow_up_meeting(state)
            state = save_to_database(state)

            # Update the session state
            active_sessions[session_id]["state"] = state

        # Return the lead data
        return {
            "lead_info": state["lead_info"],
            "follow_up_tasks": state["follow_up_tasks"],
            "conversation_summary": {
                "messages": len(state["conversation_history"]),
                "first_message_time": state["conversation_history"][0]["timestamp"] if state[
                    "conversation_history"] else None,
                "last_message_time": state["conversation_history"][-1]["timestamp"] if state[
                    "conversation_history"] else None
            }
        }

    return app


# Example usage
def run_demo():
    """
    Run a demonstration of the trade show agent
    """
    # Initialize company information
    company_info = {
        "company_name": "ABC Roasters",
        "event_name": "CMPL Mumbai Expo",
        "company_info": """
        ABC Roasters is a premium coffee supplier specializing in high-quality coffee beans for cafes, 
        restaurants, and hotels. Founded in 2010, we source our beans directly from farmers in Ethiopia, 
        Colombia, Brazil, and India. We are known for our ethical sourcing practices and consistent quality.
        """,
        "products_info": """
        Our product lineup includes:

        1. Vanilla Blend: A mix of Ethiopian & Colombian beans with vanilla notes. Perfect for espresso-based drinks.
        2. Dark Roast Blend: Specifically designed for cold brews, with chocolatey and nutty notes.
        3. Premium Single-Origin: Beans from Ethiopia (floral notes), Colombia (caramel notes), and Brazil (nutty notes).
        4. Custom Blends: Tailored to customer preferences and target flavor profiles.
        5. Coffee Equipment: Grinders, brewers, and other accessories for cafes.

        Standard MOQ is 50kg for regular orders, but we offer 5kg sample packs for first-time customers.
        Delivery times: 5-7 days for standard products, 2 weeks for custom blends.
        Pricing: We offer competitive pricing with discounts for recurring orders.
        """
    }

    # Initialize the state
    initial_state = initialize_state("demo_session", company_info)

    # Build the graph
    graph = build_trade_show_agent_graph()

    # Simulate a conversation (based on the sample in the document)
    conversation = [
        "Hello, I'm interested in learning more about your coffee products.",
        "I'm Rohan Singh, I own multiple cafes in Pune and I'm looking for coffee suppliers.",
        "I'm mainly interested in blends, but I'd also like some premium single-origin options.",
        "Yes, I can show you my business card.",
        "WhatsApp works best for me.",
        "I'd like to start with a 50kg batch to test before scaling up.",
        "No, this is my first time working with your company.",
        "I'd love to try a sample today!",
        "Yes, that would be helpful. I'd like to receive pricing information.",
        "Sounds interesting, I'd be open to discussing customized blends for my cafes."
    ]

    # Process each message
    state = initial_state

    print("=== Starting Demo Conversation ===\n")

    for message in conversation:
        print(f"User: {message}")

        # Process the message
        state = handle_user_input(state, message)

        # Run the graph for one step
        state = generate_response(state)

        # Get the assistant's response
        assistant_messages = [
            msg for msg in state["conversation_history"]
            if msg["role"] == "assistant"
        ]

        if assistant_messages:
            last_message = assistant_messages[-1]
            print(f"Assistant: {last_message['content']}")

        print(f"Current stage: {state['current_stage']}\n")

        # Update the stage
        state = update_conversation_stage(state)

    # Complete the process
    print("=== Completing Lead Processing ===\n")

    state = generate_conversation_summary(state)
    print("Generated conversation summary")

    state = create_follow_up_email(state)
    print("Created follow-up email")

    state = schedule_follow_up_meeting(state)
    print("Scheduled follow-up meeting")

    state = save_to_database(state)
    print("Saved to database")

    # Print the final state
    print("\n=== Final Lead Information ===")
    print(f"Name: {state['lead_info'].get('full_name', 'Unknown')}")
    print(f"Company: {state['lead_info'].get('company', 'Unknown')}")
    print(f"Products of Interest: {', '.join(state['lead_info'].get('products_of_interest', ['Unknown']))}")
    print(f"Interest Level: {state['lead_info'].get('interest_level', 'Unknown')}")

    print("\n=== Follow-up Tasks ===")
    for task in state["follow_up_tasks"]:
        print(f"Type: {task.get('type', 'Unknown')}")
        if task.get('type') == 'email':
            print(f"Email Content: {task.get('content', '')[:100]}...")
        else:
            print(f"Details: {task.get('details', 'No details')}")

    return state


if __name__ == "__main__":
    # Initialize a graph
    graph = build_trade_show_agent_graph()

    # Initialize state
    state = {
        "conversation_history": [],
        "lead_info": {},
        "current_stage": "greeting",
        "business_context": {
            "company_name": "Example Inc.",
            "event_name": "Demo Event"
        },
        "session_id": "test_session",
        "error": None
    }

    # Test the graph
    print("Starting conversation...")

    # First input
    state = process_input(state, "Hello, I'm interested in your products.")
    state = generate_response(state)
    print(f"AI: {state['conversation_history'][-1]['content']}")

    # Second input
    state = process_input(state, "Can you tell me more about your services?")
    state = generate_response(state)
    print(f"AI: {state['conversation_history'][-1]['content']}")

    print("Conversation completed.")
    # For actual deployment, use:
    # import uvicorn
    # app = create_api()
    # uvicorn.run(app, host="0.0.0.0", port=8000)