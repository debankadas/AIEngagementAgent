import os
import json
import logging # Keep standard logging import
from datetime import datetime
from typing import Dict, List, Optional, Any
import re  # Adding re module import at the top

# FastAPI & Pydantic
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Depends, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# LangChain & LangGraph related
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_core.messages.ai import AIMessageChunk # Import for streaming chunk
from langgraph.graph import END # Import END state marker

# Project components
from .langraph_agent import (
    initialize_session,
    determine_system_prompt, # Import helper to get initial prompt - WILL BE REPLACED LATER
    llm_instance, # Import the base LLM instance (renamed from llm)
    trade_show_agent_graph, # Keep the graph itself - WILL BE MODIFIED LATER
    LeadState,
    # Need determine_next_stage_node if we manually call it, but we are removing that
    database_tool, # For saving
    business_card_tool # Import the tool instance
)
# Import config loader
from .config import load_company_configs
# Import tools directly
from .tools.transcription import analyze_conversation_tool # For final analysis
# Import models for request/response typing if needed
from .models.lead import LeadProfile
from .models.conversation import ConversationMessage
from .logging_config import setup_logging, cleanup_old_logs # Import logging setup

# --- Setup Logging ---
# Call this early before other modules might try to log
logger = setup_logging() # Configure logging and get the root logger instance
# Get a logger specific to this module
module_logger = logging.getLogger(__name__)


# --- FastAPI App Setup ---

app = FastAPI(
    title="Trade Show Lead Management Agent API",
    description="API for interacting with the AI-powered lead engagement agent.",
    version="0.1.0"
)

# CORS Configuration (adjust origins in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows all origins for development
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods
    allow_headers=["*"], # Allows all headers
)

# --- Run Log Cleanup on Startup ---
@app.on_event("startup")
async def startup_event():
    """Run tasks when the application starts."""
    module_logger.info("Application starting up...")
    cleanup_old_logs()
    module_logger.info("Initial log cleanup complete.")

# --- In-Memory Session Storage (Replace with persistent storage in production) ---
# Stores the current state of active conversation sessions
active_sessions: Dict[str, LeadState] = {}

# --- API Request/Response Models ---

class InitSessionRequest(BaseModel):
    company_name: str = Field(..., description="Name of the exhibitor company (used for display and fallback lookup).")
    company_id: Optional[str] = Field(None, description="Optional unique identifier to directly select a company config (e.g., 'abc_roosters').")
    event_name: str = Field(..., description="Name of the trade show or event.")
    company_info: Optional[str] = Field(None, description="Optional background information about the company (overrides config if provided).")
    products_info: Optional[str] = Field(None, description="Optional information about products/services (overrides config if provided).")
    initial_prompt: Optional[str] = Field(None, description="Optional initial system message override (advanced use).")

class InitSessionResponse(BaseModel):
    session_id: str = Field(..., description="Unique ID for the newly created session.")
    status: str = Field("initialized", description="Status of the session initialization.")
    initial_message: Optional[str] = Field(None, description="Optional first message from the AI.")

class MessageRequest(BaseModel):
    session_id: str = Field(..., description="The ID of the active session.")
    message: str = Field(..., description="The message content from the user.")
    # Add image_data for business card scan if handling via HTTP
    image_data: Optional[str] = Field(None, description="Base64 encoded image data (for business card scan).")

class MessageResponse(BaseModel):
    response: str = Field(..., description="The AI assistant's response message.")
    current_stage: str = Field(..., description="The current stage of the conversation.")
    session_id: str = Field(..., description="The session ID.")
    # Optionally include updated lead profile snippet

class ExportLeadResponse(BaseModel):
    session_id: str
    lead_profile: Optional[Dict[str, Any]] = Field(None, description="Extracted lead profile data.")
    conversation_summary: Optional[Dict[str, Any]] = Field(None, description="Summary metrics of the conversation.")
    status: str = "exported"

# --- Helper Function ---

def get_session_state(session_id: str) -> LeadState:
    """Retrieves the state for a given session ID, raising HTTPException if not found."""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    return active_sessions[session_id]

# --- API Endpoints ---

@app.get("/api/companies")
async def get_companies_endpoint():
    """
    Retrieves a list of all available companies from the configuration file.
    """
    try:
        all_company_configs = load_company_configs()
        company_list = []
        
        for company_id, config in all_company_configs.items():
            company_list.append({
                "id": company_id,
                "display_name": config.get("display_name", company_id),
                "info": config.get("company_info", ""),
                "products_info": config.get("products_info", ""),
                "product_varieties": config.get("product_varieties", []),
                "company_type": config.get("company_type", "product"),
                "field_labels": config.get("field_labels", {})
            })
        
        return {"companies": company_list}
    except Exception as e:
        module_logger.error(f"Error retrieving companies: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve companies: {e}")

@app.post("/api/init-session", response_model=InitSessionResponse)
async def init_session_endpoint(request: InitSessionRequest):
    """
    Initializes a new conversation session for the trade show agent.
    """
    session_id = f"session_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"

    # --- Load Company Configuration ---
    all_company_configs = load_company_configs()
    company_key = "default" # Start with default

    # Try finding by ID first
    if request.company_id and request.company_id in all_company_configs:
        company_key = request.company_id
        module_logger.info(f"Found company config using provided ID: '{company_key}'")
    else:
        # Try finding by name (simple match for now, could be more robust)
        normalized_request_name = request.company_name.lower().replace(" ", "_") # Basic normalization
        found_by_name = False
        for key in all_company_configs:
             # Allow matching normalized name or display name (case-insensitive)
             if key == normalized_request_name or \
                all_company_configs[key].get('display_name', '').lower() == request.company_name.lower():
                 company_key = key
                 found_by_name = True
                 module_logger.info(f"Found company config by matching name '{request.company_name}' to key: '{company_key}'")
                 break
        if not found_by_name and request.company_id:
             module_logger.warning(f"Provided company_id '{request.company_id}' not found in configs. Falling back to 'default'.")
        elif not found_by_name:
             module_logger.warning(f"Could not find company config matching name '{request.company_name}'. Falling back to 'default'.")

    company_config = all_company_configs.get(company_key, all_company_configs["default"]) # Ensure we always get a config

    # --- Prepare Business Context (can be overridden by request) ---
    business_context = {
        "company_name": company_config.get("display_name", request.company_name), # Use display name from config
        "event_name": request.event_name,
        "company_info": request.company_info or company_config.get("company_info", "N/A"), # Request overrides config
        "products_info": request.products_info or company_config.get("products_info", "N/A") # Request overrides config
    }
    
    try:
        # Initialize state using the function from langraph_agent
        state = initialize_session(session_id, business_context)

        # --- Inject Company Config and Initialize Tracking ---
        state["company_config"] = company_config
        state["collected_fields"] = {} # Initialize empty dict for collected data
        module_logger.info(f"Injected config for '{company_key}' into session state.")

        # Handle optional initial prompt override (less common with dynamic prompts)
        if request.initial_prompt:
             state["conversation_history"].insert(0, SystemMessage(content=request.initial_prompt))
             module_logger.info("Using provided initial system prompt override.")

        active_sessions[session_id] = state
        module_logger.info(f"Initialized session: {session_id}")

        # Generate the initial greeting message directly
        # --- Generate Initial Greeting ---
        # TODO: This part will be replaced/modified significantly when agent logic changes.
        # For now, we still use the old determine_system_prompt for the *initial* greeting,
        # but the dynamic logic will take over on subsequent turns.
        # We might use company_config['initial_greeting_prompt'] here later.
        initial_system_prompt = determine_system_prompt(state) # Uses old logic for now
        config = {"configurable": {"session_id": session_id}, "recursion_limit": 50}
        initial_response = llm_instance.invoke([SystemMessage(content=initial_system_prompt)], config=config)

        if isinstance(initial_response, BaseMessage):
             state["conversation_history"].append(initial_response)
             initial_ai_message_content = initial_response.content
        else:
             initial_ai_message_content = "Welcome!"
             state["conversation_history"].append(AIMessage(content=initial_ai_message_content))

        active_sessions[session_id] = state # Store state with greeting
        module_logger.info(f"Initialized session: {session_id} with initial greeting.")
        return InitSessionResponse(session_id=session_id, initial_message=initial_ai_message_content)

    except Exception as e:
        module_logger.error(f"Error initializing session {session_id}: {e}", exc_info=True) # Log exception info
        raise HTTPException(status_code=500, detail=f"Failed to initialize session: {e}")


@app.post("/api/send-message", response_model=MessageResponse)
async def send_message_endpoint(request: MessageRequest):
    """
    Sends a message from the user to the specified session and gets the AI response.
    Handles potential business card image data.
    """
    state = get_session_state(request.session_id)
    # Base config for the session
    config = {"configurable": {"session_id": request.session_id}}
    # Config with increased recursion limit for the graph invoke call
    config_with_limit = {**config, "recursion_limit": 10}

    # Get the current state before appending the new message
    collected_fields = state.get("collected_fields", {})
    
    # Add the user's message to the conversation history
    state["conversation_history"].append(HumanMessage(content=request.message))
    state["input"] = request.message
    
    # Process user response to update collected fields based on context
    if len(state["conversation_history"]) >= 2:
        # Get the last AI message to understand the context of the user's current response
        last_ai_messages = [msg for msg in reversed(state["conversation_history"]) if isinstance(msg, AIMessage)]
        if last_ai_messages:
            last_ai_message = last_ai_messages[0].content.lower()
            current_message = request.message.lower()
            
            # Update product variety selection
            if "product varieties" in last_ai_message or "which of our product" in last_ai_message:
                product_options = ["starter feed", "grower feed", "layer feed", "broiler feed", "custom formulations"]
                for product in product_options:
                    if product in current_message:
                        collected_fields["preferred_product"] = product.title()
                        break
                if "preferred_product" not in collected_fields and current_message.strip():
                    # If we couldn't match a specific product but user responded something
                    collected_fields["preferred_product"] = current_message.strip()
            
            # Get company type to determine how to process the fields
            company_type = state.get("company_config", {}).get("company_type", "product")
            field_labels = state.get("company_config", {}).get("field_labels", {})
            
            # Handle order volume (for product companies) or engagement scope (for service companies)
            order_volume_label = field_labels.get("order_volume", "Minimum Order Quantity (MOQ)").lower()
            if company_type == "product" and (order_volume_label in last_ai_message.lower() or "moq" in last_ai_message) and "preferred" not in last_ai_message:
                # Extract numbers from the response for order volume/MOQ
                numbers = re.findall(r'\d+', current_message)
                if numbers:
                    quantity = numbers[0]  # Get the first number
                    units = ""
                    # Check for common units
                    if "kg" in current_message or "kilo" in current_message:
                        units = "kg"
                    elif "ton" in current_message:
                        units = "tons"
                    elif "lb" in current_message or "pound" in current_message:
                        units = "lbs"
                    
                    collected_fields["order_volume"] = f"{quantity}{units}"
                else:
                    # If no specific number found, store the general response
                    collected_fields["order_volume"] = current_message
            
            # Handle engagement scope for service companies
            engagement_scope_label = field_labels.get("engagement_scope", "Project Scale/Scope").lower()
            if company_type == "service" and engagement_scope_label in last_ai_message.lower():
                # For service companies, we care about different metrics
                if "enterprise" in current_message.lower():
                    collected_fields["engagement_scope"] = "Enterprise"
                elif "department" in current_message.lower() or "team" in current_message.lower():
                    collected_fields["engagement_scope"] = "Department/Team"
                elif "pilot" in current_message.lower() or "poc" in current_message.lower() or "proof of concept" in current_message.lower():
                    collected_fields["engagement_scope"] = "Pilot/PoC"
                elif any(scale in current_message.lower() for scale in ["small", "limited", "basic"]):
                    collected_fields["engagement_scope"] = "Small-Scale"
                elif any(scale in current_message.lower() for scale in ["medium", "moderate"]):
                    collected_fields["engagement_scope"] = "Medium-Scale"
                elif any(scale in current_message.lower() for scale in ["large", "extensive", "full"]):
                    collected_fields["engagement_scope"] = "Large-Scale"
                else:
                    # Store the raw response if we can't categorize it
                    collected_fields["engagement_scope"] = current_message
            
            # Update farm size based on response (specific to agriculture companies)
            if "farm size" in last_ai_message or "size of your farm" in last_ai_message:
                # Extract numbers and units for farm size
                numbers = re.findall(r'\d+', current_message)
                if numbers:
                    size = numbers[0]  # Get the first number
                    units = ""
                    # Check for common units
                    if "acre" in current_message:
                        units = "acres"
                    elif "hectare" in current_message:
                        units = "hectares"
                    
                    collected_fields["farm_size"] = f"{size} {units}".strip()
                else:
                    collected_fields["farm_size"] = current_message
            
            # Update current supplier
            if "current supplier" in last_ai_message:
                if "no supplier" in current_message or "none" in current_message:
                    collected_fields["current_supplier"] = "None"
                else:
                    collected_fields["current_supplier"] = current_message
            
            # Update preferred communication based on response
            if "preferred method of communication" in last_ai_message or "preferred communication" in last_ai_message:
                if "whatsapp" in current_message:
                    collected_fields["preferred_communication"] = "WhatsApp"
                elif "email" in current_message:
                    collected_fields["preferred_communication"] = "Email"
                elif "phone" in current_message or "call" in current_message:
                    collected_fields["preferred_communication"] = "Phone"
            
            # Update custom interest based on response
            if "custom blend" in last_ai_message or "custom solution" in last_ai_message:
                if any(word in current_message for word in ["yes", "yeah", "sure", "interested"]):
                    collected_fields["custom_interest"] = "Yes"
                elif any(word in current_message for word in ["no", "nope", "not"]):
                    collected_fields["custom_interest"] = "No"
            
            # Update pricing catalog request
            if "pricing catalog" in last_ai_message:
                if any(word in current_message for word in ["yes", "yeah", "sure", "interested"]):
                    collected_fields["pricing_catalog_request"] = "Yes"
                elif any(word in current_message for word in ["no", "nope", "not"]):
                    collected_fields["pricing_catalog_request"] = "No"
            
            # Handle sample or demo requests based on company type
            if company_type == "product" and "sample" in last_ai_message and ("tasting" in last_ai_message or "demo" in last_ai_message):
                if any(word in current_message for word in ["yes", "yeah", "sure", "interested"]):
                    collected_fields["sample_tasting_request"] = "Yes"
                elif any(word in current_message for word in ["no", "nope", "not"]):
                    collected_fields["sample_tasting_request"] = "No"
            elif company_type == "service" and ("demo" in last_ai_message or "demonstration" in last_ai_message or "technical demo" in last_ai_message):
                if any(word in current_message for word in ["yes", "yeah", "sure", "interested"]):
                    collected_fields["demo_request"] = "Yes"
                elif any(word in current_message for word in ["no", "nope", "not"]):
                    collected_fields["demo_request"] = "No"
            
            # Update early access interest
            if "early access" in last_ai_message:
                if any(word in current_message for word in ["yes", "yeah", "sure", "interested"]):
                    collected_fields["early_access_interest"] = "Yes"
                elif any(word in current_message for word in ["no", "nope", "not"]):
                    collected_fields["early_access_interest"] = "No"
            
            # Update customer objections
            if "concern" in last_ai_message or "objection" in last_ai_message:
                if any(word in current_message for word in ["yes", "yeah", "concern", "issue", "problem"]):
                    collected_fields["customer_objections"] = current_message
                elif any(word in current_message for word in ["no", "nope", "not", "none"]):
                    collected_fields["customer_objections"] = "None"
    
    # Update the state with any collected fields
    state["collected_fields"] = collected_fields
    
    # --- Handle Image Data ---
    # If image data is provided, store it in the state for the agent/tool node to use later.
    # Also add a placeholder message to history so the agent knows an image was provided.
    if request.image_data:
        module_logger.info(f"Received image data for session {request.session_id}. Storing in state.")
        state["pending_image_data"] = request.image_data
        # DO NOT add a message to history here. The agent will decide to scan based on missing info.
        # The image data is stored in state and injected into the tool call later.
    else:
        # Ensure pending_image_data is None if no image was sent in this request
        # Prevents using stale image data from a previous turn if the agent calls the tool unexpectedly.
        state["pending_image_data"] = None

    # Invoke the graph with increased recursion limit
    try:
        updated_state = trade_show_agent_graph.invoke(state, config_with_limit) # Use config_with_limit
        active_sessions[request.session_id] = updated_state
    except Exception as e:
        module_logger.error(f"Error invoking graph for session {request.session_id}: {e}", exc_info=True)
        last_ai_message = next((msg.content for msg in reversed(state["conversation_history"]) if isinstance(msg, AIMessage)), "An error occurred processing your request.")
        # Include the specific error in the detail for better debugging
        raise HTTPException(status_code=500, detail=f"Agent processing error: {str(e)}")

    ai_response_content = next((msg.content for msg in reversed(updated_state["conversation_history"]) if isinstance(msg, AIMessage)), "...")

    return MessageResponse(
        response=ai_response_content,
        current_stage=updated_state["current_stage"],
        session_id=request.session_id
    )


# --- Conversation History Endpoints ---

@app.get("/api/sessions")
async def get_sessions_endpoint(limit: int = 50):
    """
    Retrieves a list of recent conversation session summaries.
    """
    if not database_tool.enabled:
        raise HTTPException(status_code=503, detail="Database feature is disabled.")
    try:
        sessions_json = database_tool.get_conversation_sessions(limit=limit)
        sessions_data = json.loads(sessions_json)
        if sessions_data.get("status") == "error":
             raise HTTPException(status_code=500, detail=sessions_data.get("message", "Error retrieving sessions."))
        # Return the list of sessions directly
        return sessions_data.get("sessions", [])
    except ConnectionError as ce:
         raise HTTPException(status_code=500, detail=f"Database connection error: {ce}")
    except Exception as e:
        module_logger.error(f"Error in /api/sessions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error retrieving sessions: {e}")


@app.get("/api/conversations/{conversation_id}")
async def get_conversation_messages_endpoint(conversation_id: str):
    """
    Retrieves the full message history for a specific conversation ID.
    """
    if not database_tool.enabled:
        raise HTTPException(status_code=503, detail="Database feature is disabled.")
    try:
        messages_json = database_tool.get_conversation_messages(conversation_id=conversation_id)
        messages_data = json.loads(messages_json)

        status = messages_data.get("status")
        if status == "error":
            raise HTTPException(status_code=500, detail=messages_data.get("message", "Error retrieving conversation messages."))
        elif status == "not_found":
            raise HTTPException(status_code=404, detail=messages_data.get("message", f"Conversation {conversation_id} not found."))
        elif status == "success":
            # Return the conversation data directly
            return messages_data.get("conversation", {})
        else:
             # Should not happen based on tool implementation, but handle defensively
             raise HTTPException(status_code=500, detail="Unexpected response status from database tool.")

    except ConnectionError as ce:
         raise HTTPException(status_code=500, detail=f"Database connection error: {ce}")
    except Exception as e:
        module_logger.error(f"Error in /api/conversations/{conversation_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error retrieving conversation: {e}")


# --- Export Transcript Endpoint ---

@app.post("/api/export-transcript/{session_id}")
async def export_transcript_endpoint(session_id: str):
    """
    Exports the full conversation transcript for a given session ID.
    """
    module_logger.info(f"Received request to export transcript for session: {session_id}")
    try:
        state = get_session_state(session_id)
        history = state.get("conversation_history", [])

        if not history:
            module_logger.warning(f"No conversation history found for session {session_id}")
            raise HTTPException(status_code=404, detail="No conversation history found for this session.")

        # Format the transcript (simple text format)
        transcript_lines = []
        for msg in history:
            role = "Unknown"
            if isinstance(msg, HumanMessage):
                role = "Human"
            elif isinstance(msg, AIMessage):
                role = "Assistant"
            elif isinstance(msg, SystemMessage):
                role = "System"
            elif isinstance(msg, ToolMessage):
                role = f"Tool ({msg.name})"

            timestamp = getattr(msg, 'timestamp', datetime.now()).strftime('%Y-%m-%d %H:%M:%S') # Add timestamp if available
            content = getattr(msg, 'content', '[No Content]')
            transcript_lines.append(f"[{timestamp}] {role}: {content}")

        transcript_text = "\n".join(transcript_lines)
        
        module_logger.info(f"Successfully generated transcript for session {session_id}")
        # Return as plain text - frontend will handle download
        return {"transcript": transcript_text, "status": "success"}

    except HTTPException as http_exc:
        # Re-raise HTTPExceptions directly (like 404 Not Found)
        raise http_exc
    except Exception as e:
        module_logger.error(f"Error exporting transcript for session {session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to export transcript: {e}")



@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """Handles real-time conversation via WebSocket (using ainvoke)."""
    try:
        state = get_session_state(session_id)
    except HTTPException:
        await websocket.accept()
        await websocket.send_json({"type": "error", "payload": {"message": f"Session '{session_id}' not found."}})
        await websocket.close(code=1008)
        return

    await websocket.accept()
    module_logger.info(f"WebSocket connected for session: {session_id}")
    # Config with increased recursion limit for ainvoke
    config_with_limit = {"configurable": {"session_id": session_id}, "recursion_limit": 50}

    try:
        while True:
            data_json = await websocket.receive_json()
            module_logger.debug(f"WS Received JSON (Session {session_id}): {data_json}") # Use debug level for verbose messages

            message_type = data_json.get("type")
            message_payload = data_json.get("payload", {})

            # --- Simplified Input Handling ---
            current_stage = state.get("current_stage")

            if message_type == "text" and "content" in message_payload:
                # Handle text input
                user_message_content = message_payload["content"]
                
                # Get the current collected fields before updating
                collected_fields = state.get("collected_fields", {})
                
                # Add the message to conversation history
                state["conversation_history"].append(HumanMessage(content=user_message_content))
                state["input"] = user_message_content
                
                # Process user response to update collected fields based on context
                if len(state["conversation_history"]) >= 2:
                    # Get the last AI message to understand the context of the user's current response
                    last_ai_messages = [msg for msg in reversed(state["conversation_history"]) if isinstance(msg, AIMessage)]
                    if last_ai_messages:
                        last_ai_message = last_ai_messages[0].content.lower()
                        current_message = user_message_content.lower()
                        
                        # Update product variety selection
                        if "product varieties" in last_ai_message or "which of our product" in last_ai_message:
                            product_options = ["starter feed", "grower feed", "layer feed", "broiler feed", "custom formulations"]
                            for product in product_options:
                                if product in current_message:
                                    collected_fields["preferred_product"] = product.title()
                                    break
                            if "preferred_product" not in collected_fields and current_message.strip():
                                # If we couldn't match a specific product but user responded something
                                collected_fields["preferred_product"] = current_message.strip()
                        
                        # Get company type to determine how to process the fields
                        company_type = state.get("company_config", {}).get("company_type", "product")
                        field_labels = state.get("company_config", {}).get("field_labels", {})
                        
                        # Handle order volume (for product companies) or engagement scope (for service companies)
                        order_volume_label = field_labels.get("order_volume", "Minimum Order Quantity (MOQ)").lower()
                        if company_type == "product" and (order_volume_label in last_ai_message.lower() or "moq" in last_ai_message) and "preferred" not in last_ai_message:
                            # Extract numbers from the response for order volume/MOQ
                            numbers = re.findall(r'\d+', current_message)
                            if numbers:
                                quantity = numbers[0]  # Get the first number
                                units = ""
                                # Check for common units
                                if "kg" in current_message or "kilo" in current_message:
                                    units = "kg"
                                elif "ton" in current_message:
                                    units = "tons"
                                elif "lb" in current_message or "pound" in current_message:
                                    units = "lbs"
                                
                                collected_fields["order_volume"] = f"{quantity}{units}"
                            else:
                                # If no specific number found, store the general response
                                collected_fields["order_volume"] = current_message
                        
                        # Handle engagement scope for service companies
                        engagement_scope_label = field_labels.get("engagement_scope", "Project Scale/Scope").lower()
                        if company_type == "service" and engagement_scope_label in last_ai_message.lower():
                            # For service companies, we care about different metrics
                            if "enterprise" in current_message.lower():
                                collected_fields["engagement_scope"] = "Enterprise"
                            elif "department" in current_message.lower() or "team" in current_message.lower():
                                collected_fields["engagement_scope"] = "Department/Team"
                            elif "pilot" in current_message.lower() or "poc" in current_message.lower() or "proof of concept" in current_message.lower():
                                collected_fields["engagement_scope"] = "Pilot/PoC"
                            elif any(scale in current_message.lower() for scale in ["small", "limited", "basic"]):
                                collected_fields["engagement_scope"] = "Small-Scale"
                            elif any(scale in current_message.lower() for scale in ["medium", "moderate"]):
                                collected_fields["engagement_scope"] = "Medium-Scale"
                            elif any(scale in current_message.lower() for scale in ["large", "extensive", "full"]):
                                collected_fields["engagement_scope"] = "Large-Scale"
                            else:
                                # Store the raw response if we can't categorize it
                                collected_fields["engagement_scope"] = current_message
                        
                        # Update farm size based on response
                        if "farm size" in last_ai_message or "size of your farm" in last_ai_message:
                            # Extract numbers and units for farm size
                            numbers = re.findall(r'\d+', current_message)
                            if numbers:
                                size = numbers[0]  # Get the first number
                                units = ""
                                # Check for common units
                                if "acre" in current_message:
                                    units = "acres"
                                elif "hectare" in current_message:
                                    units = "hectares"
                                
                                collected_fields["farm_size"] = f"{size} {units}".strip()
                            else:
                                collected_fields["farm_size"] = current_message
                        
                        # Update current supplier
                        if "current supplier" in last_ai_message:
                            if "no supplier" in current_message or "none" in current_message:
                                collected_fields["current_supplier"] = "None"
                            else:
                                collected_fields["current_supplier"] = current_message
                        
                        # Update preferred communication based on response
                        if "preferred method of communication" in last_ai_message or "preferred communication" in last_ai_message:
                            if "whatsapp" in current_message:
                                collected_fields["preferred_communication"] = "WhatsApp"
                            elif "email" in current_message:
                                collected_fields["preferred_communication"] = "Email"
                            elif "phone" in current_message or "call" in current_message:
                                collected_fields["preferred_communication"] = "Phone"
                        
                        # Update custom interest based on response
                        if "custom blend" in last_ai_message or "custom solution" in last_ai_message:
                            if any(word in current_message for word in ["yes", "yeah", "sure", "interested"]):
                                collected_fields["custom_interest"] = "Yes"
                            elif any(word in current_message for word in ["no", "nope", "not"]):
                                collected_fields["custom_interest"] = "No"
                        
                        # Update pricing catalog request
                        if "pricing catalog" in last_ai_message:
                            if any(word in current_message for word in ["yes", "yeah", "sure", "interested"]):
                                collected_fields["pricing_catalog_request"] = "Yes"
                            elif any(word in current_message for word in ["no", "nope", "not"]):
                                collected_fields["pricing_catalog_request"] = "No"
                        
                        # Handle sample or demo requests based on company type
                        if company_type == "product" and "sample" in last_ai_message and ("tasting" in last_ai_message or "demo" in last_ai_message):
                            if any(word in current_message for word in ["yes", "yeah", "sure", "interested"]):
                                collected_fields["sample_tasting_request"] = "Yes"
                            elif any(word in current_message for word in ["no", "nope", "not"]):
                                collected_fields["sample_tasting_request"] = "No"
                        elif company_type == "service" and ("demo" in last_ai_message or "demonstration" in last_ai_message or "technical demo" in last_ai_message):
                            if any(word in current_message for word in ["yes", "yeah", "sure", "interested"]):
                                collected_fields["demo_request"] = "Yes"
                            elif any(word in current_message for word in ["no", "nope", "not"]):
                                collected_fields["demo_request"] = "No"
                        
                        # Update early access interest
                        if "early access" in last_ai_message:
                            if any(word in current_message for word in ["yes", "yeah", "sure", "interested"]):
                                collected_fields["early_access_interest"] = "Yes"
                            elif any(word in current_message for word in ["no", "nope", "not"]):
                                collected_fields["early_access_interest"] = "No"
                        
                        # Update customer objections
                        if "concern" in last_ai_message or "objection" in last_ai_message:
                            if any(word in current_message for word in ["yes", "yeah", "concern", "issue", "problem"]):
                                collected_fields["customer_objections"] = current_message
                            elif any(word in current_message for word in ["no", "nope", "not", "none"]):
                                collected_fields["customer_objections"] = "None"
                
                # Update the state with the collected fields
                state["collected_fields"] = collected_fields
                
            elif current_stage == "business_card_scan" and message_type == "image" and "image_data" in message_payload:
                # Handle image input ONLY if in the correct stage
                module_logger.info(f"Received image data for business card scan (Session {session_id}).")
                image_data = message_payload["image_data"]
                try:
                    scan_result_json = business_card_tool.scan_business_card.invoke({"image_data": image_data})
                    # Add ToolMessage directly to state history
                    state["conversation_history"].append(ToolMessage(content=scan_result_json, tool_call_id="websocket_scan_call"))
                    module_logger.info(f"Scan tool result added to history for session {session_id}")
                    # The graph will be invoked next and process this ToolMessage
                except Exception as tool_err:
                    module_logger.error(f"Error during direct business card scan for session {session_id}: {tool_err}", exc_info=True)
                    await websocket.send_json({"type": "error", "payload": {"message": f"Error scanning card: {tool_err}"}})
                    continue # Skip graph invocation on tool error
            else:
                # Received unexpected message type or format for the current stage
                module_logger.warning(f"Received unexpected message format '{message_type}' for stage '{current_stage}' in session {session_id}.")
                expected_input = "business card image" if current_stage == "business_card_scan" else "text message"
                await websocket.send_json({"type": "error", "payload": {"message": f"Invalid input. Expected {expected_input} for stage '{current_stage}'."}}) # Added closing characters and stage info

    # FIX: Add except block for WebSocket disconnection
    except WebSocketDisconnect:
        module_logger.info(f"WebSocket disconnected for session: {session_id}")
        # Add any specific cleanup needed on disconnect here

    # FIX: Add generic except block for other errors
    except Exception as e:
        module_logger.error(f"An error occurred in WebSocket handler for session {session_id}: {e}", exc_info=True) # Completed error message
        try:
            # Try to inform the client about the error
            await websocket.send_json({"type": "error", "payload": {"message": "An internal server error occurred."}})
        except Exception as send_error:
            module_logger.error(f"Failed to send error message to WebSocket for session {session_id}: {send_error}")

    # FIX: Add finally block for cleanup
    finally:
        module_logger.info(f"Closing WebSocket connection handler for session: {session_id}")
        # WebSocket closing is often handled by the framework (FastAPI/Starlette),
        # but add any other necessary resource cleanup here.
