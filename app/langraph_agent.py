import os
import json
from typing import Dict, List, Optional, Any, TypedDict, Annotated, Sequence
from typing_extensions import TypedDict
import logging
from datetime import datetime

# Get a logger for this module
logger = logging.getLogger(__name__)
# LangChain & LangGraph Core
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition # Re-import tools_condition
from langgraph.checkpoint.base import BaseCheckpointSaver # For potential future use

# Project components
from .models.lead import LeadProfile, BusinessCardInfo
from .models.conversation import ConversationMessage, ConversationHistory
from .tools.business_card import BusinessCardTool
from .tools.database import DatabaseTool
from .tools.email import EmailTool
from .tools.scheduling import SchedulingTool
# Import the actual tool objects now
# from .tools.transcription import analyze_conversation_tool # Keep commented if not used directly by agent node
from .utils import prompts
from .utils.helpers import format_lead_info_summary, format_follow_up_summary, parse_tool_calls
# Import config reader
from .config import get_llm_config

# --- State Definition ---

class LeadState(TypedDict):
    """Represents the state of the conversation and lead data."""
    input: Optional[str] # Last user input
    output: Optional[str] # Last AI output (text part)
    conversation_history: List[BaseMessage] # Full history
    current_stage: str # Current stage (will be less important later)
    stage_history: List[str] # History of stages visited
    lead_profile: Optional[Dict[str, Any]] # Data extracted about the lead
    business_context: Dict[str, Any] # Info about the event/company running the agent
    session_id: str # Unique ID for the session
    tool_calls: Optional[List[Dict[str, Any]]] # Pending tool calls from LLM
    error: Optional[str] # Any errors encountered
    pending_image_data: Optional[str] # Holds base64 image data if provided by user, waiting for tool call

    # --- New fields for dynamic flow ---
    company_config: Optional[Dict[str, Any]] # Loaded config for the specific company
    collected_fields: Dict[str, Any] # Fields collected so far (key: field_name, value: collected_value)
    scan_just_processed: Optional[bool] # Flag to indicate a scan was just processed in the previous turn


# --- Tool Instantiation ---

business_card_tool = BusinessCardTool()
database_tool = DatabaseTool()
email_tool = EmailTool()
scheduling_tool = SchedulingTool()
# transcription_tool = TranscriptionTool() # Keep commented if not used

# Instantiate the BusinessCardTool
business_card_tool = BusinessCardTool()
database_tool = DatabaseTool() # Assuming these might be used later or by other parts
email_tool = EmailTool()
scheduling_tool = SchedulingTool()

# Add the business card tool to the list available to the agent
agent_tools: List[BaseTool] = [business_card_tool]
logger.info(f"Agent initialized with tools: {[tool.name for tool in agent_tools]}")

# --- LLM and Agent Setup ---

def get_llm():
    """Initializes the LLM based on config.yaml settings."""
    config = get_llm_config()
    primary_provider = config.get("primary_provider", "openai") # Default to openai if not set
    anthropic_model = config.get("anthropic_model", "claude-3-haiku-20240307")
    openai_model = config.get("openai_model", "gpt-4o-mini")
    temperature = config.get("temperature", 0.1)

    llm_instance = None
    primary_error = None
    fallback_error = None

    anthropic_key_present = bool(os.getenv("ANTHROPIC_API_KEY"))
    openai_key_present = bool(os.getenv("OPENAI_API_KEY"))

    logger.info(f"Attempting to initialize LLM. Primary provider from config: '{primary_provider}'")

    # Try primary provider first
    if primary_provider == "anthropic":
        if anthropic_key_present:
            try:
                llm_instance = ChatAnthropic(model=anthropic_model, temperature=temperature)
                logger.info(f"Successfully initialized primary LLM: Anthropic '{anthropic_model}'")
            except Exception as e:
                primary_error = f"Failed to initialize primary Anthropic model '{anthropic_model}': {e}"
                logger.warning(f"{primary_error}", exc_info=True)
        else:
            primary_error = "Anthropic is primary provider, but ANTHROPIC_API_KEY is not set."
            logger.warning(f"{primary_error}")
    elif primary_provider == "openai":
        if openai_key_present:
            try:
                llm_instance = ChatOpenAI(model=openai_model, temperature=temperature)
                logger.info(f"Successfully initialized primary LLM: OpenAI '{openai_model}'")
            except Exception as e:
                primary_error = f"Failed to initialize primary OpenAI model '{openai_model}': {e}"
                logger.warning(f"{primary_error}", exc_info=True)
        else:
            primary_error = "OpenAI is primary provider, but OPENAI_API_KEY is not set."
            logger.warning(f"{primary_error}")
    else:
        primary_error = f"Unknown primary_provider '{primary_provider}' in config.yaml."
        logger.error(f"{primary_error}")


    # Try fallback provider if primary failed
    if not llm_instance:
        logger.warning("Primary LLM initialization failed or key missing, attempting fallback...")
        if primary_provider == "anthropic": # Fallback is OpenAI
            if openai_key_present:
                try:
                    llm_instance = ChatOpenAI(model=openai_model, temperature=temperature)
                    logger.info(f"Successfully initialized fallback LLM: OpenAI '{openai_model}'")
                except Exception as e:
                    fallback_error = f"Failed to initialize fallback OpenAI model '{openai_model}': {e}"
                    logger.warning(f"{fallback_error}", exc_info=True)
            else:
                 fallback_error = "Fallback OpenAI selected, but OPENAI_API_KEY is not set."
                 logger.warning(f"{fallback_error}")
        elif primary_provider == "openai": # Fallback is Anthropic
            if anthropic_key_present:
                try:
                    llm_instance = ChatAnthropic(model=anthropic_model, temperature=temperature)
                    logger.info(f"Successfully initialized fallback LLM: Anthropic '{anthropic_model}'")
                except Exception as e:
                    fallback_error = f"Failed to initialize fallback Anthropic model '{anthropic_model}': {e}"
                    logger.warning(f"{fallback_error}", exc_info=True)
            else:
                 fallback_error = "Fallback Anthropic selected, but ANTHROPIC_API_KEY is not set."
                 logger.warning(f"{fallback_error}")

    # Final check
    if not llm_instance:
        error_message = f"Fatal: No LLM could be initialized. Primary Error: {primary_error}. Fallback Error: {fallback_error}."
        logger.critical(error_message) # Use critical for fatal errors before raising
        raise ConnectionError(error_message)

    return llm_instance

# Initialize LLM based on config and bind tools
llm_instance = get_llm()
llm_with_tools = llm_instance.bind_tools(agent_tools)

# --- Graph Nodes ---

def determine_system_prompt(state: LeadState) -> str:
    stage = state["current_stage"]
    context = state["business_context"]
    lead_summary = format_lead_info_summary(state.get("lead_profile"))
    # Updated prompt map with new stages
    prompt_map = {
        "greeting": prompts.GREETING_PROMPT,
        "ask_preference": prompts.ASK_PREFERENCE_PROMPT,
        "ask_card": prompts.ASK_CARD_PROMPT,
        "business_card_scan": prompts.BUSINESS_CARD_SCAN_PROMPT,
        "confirm_scan_details": prompts.CONFIRM_SCAN_DETAILS_PROMPT,
        "ask_contact_preference": prompts.ASK_CONTACT_PREFERENCE_PROMPT,
        "ask_moq": prompts.ASK_MOQ_PROMPT,
        "ask_custom_interest": prompts.ASK_CUSTOM_INTEREST_PROMPT,
        "ask_catalog_request": prompts.ASK_CATALOG_REQUEST_PROMPT,
        "ask_sample_request": prompts.ASK_SAMPLE_REQUEST_PROMPT,
        "ask_early_access": prompts.ASK_EARLY_ACCESS_PROMPT,
        "ask_concerns": prompts.ASK_CONCERNS_PROMPT,
        "product_discussion": prompts.PRODUCT_DISCUSSION_PROMPT,
        "objection_handling": prompts.OBJECTION_HANDLING_PROMPT,
        "next_steps": prompts.NEXT_STEPS_PROMPT,
        "closing": prompts.CLOSING_PROMPT,
    }
    # Use a more specific default or raise error if stage is unknown
    default_prompt = prompts.GREETING_PROMPT # Defaulting to greeting if stage unknown
    template = prompt_map.get(stage, default_prompt)
    if not template:
         # Handle cases where the stage might be invalid
         logger.error(f"Unknown stage '{stage}' encountered in determine_system_prompt.")
         # Fallback or raise an error - using ask_preference as a safe default for now
         template = default_prompt

    # Prepare context, adding lead details if available for specific prompts
    lead_profile = state.get("lead_profile", {})
    # Get product varieties from company config
    company_config = state.get("company_config", {})
    product_varieties = company_config.get('product_varieties', [])
    product_varieties_str = ", ".join(product_varieties) if product_varieties else "N/A"
    
    # Get collected fields for preference and next steps prompts
    collected_fields = state.get("collected_fields", {})
    
    context_with_defaults = {
        "company_name": context.get("company_name", "Our Company"),
        "event_name": context.get("event_name", "the Event"),
        "company_info": context.get("company_info", "No company info provided."),
        "products_info": context.get("products_info", "No product info provided."),
        "product_varieties_info": product_varieties_str,
        "lead_info_summary": lead_summary,
        "follow_up_summary": "Not yet determined.", # Assuming this is always needed or handled
        # Add specific lead details for prompts that need them
        "lead_name": lead_profile.get("full_name", "Visitor"),
        "lead_company": lead_profile.get("company", "their company"),
        "lead_location": lead_profile.get("location", "their location"),
        # Add collected fields for references in prompts
        "preferred_product": collected_fields.get("preferred_product", "products"),
        "moq": collected_fields.get("moq", ""),
        "preferred_communication": collected_fields.get("preferred_communication", ""),
        "custom_interest": collected_fields.get("custom_interest", ""),
        "pricing_catalog_request": collected_fields.get("pricing_catalog_request", ""),
        "sample_tasting_request": collected_fields.get("sample_tasting_request", ""),
        "early_access_interest": collected_fields.get("early_access_interest", ""),
    }

    try:
        return template.format(**context_with_defaults)
    except KeyError as e:
        logger.error(f"Error formatting prompt for stage '{stage}': Missing key {e}", exc_info=True)
        # Return a generic safe prompt or raise error
        return "Hello! How can I help you today?"
    # Removed duplicated lines from previous version here


def generate_dynamic_prompt(state: LeadState) -> str:
    """Generates a dynamic system prompt based on company config and conversation state."""
    logger.debug("--- Generating Dynamic Prompt ---")
    company_config = state.get("company_config")
    collected_fields = state.get("collected_fields", {})
    business_context = state.get("business_context", {})
    conversation_history = state.get("conversation_history", [])
    scan_just_processed = state.get("scan_just_processed", False) # Check the flag

    if not company_config:
        logger.error("Company config missing in state! Falling back to generic prompt.")
        # Fallback to a very basic prompt if config is somehow missing
        return "You are a helpful assistant. Please engage the user in conversation."

    # --- Identify Missing Fields ---
    required = company_config.get("required_fields", [])
    optional = company_config.get("optional_fields", [])
    missing_required = [field for field in required if field not in collected_fields]
    missing_optional = [field for field in optional if field not in collected_fields]
    
    # --- Determine Question Priority ---
    # Define the sequence of questions to follow to create a natural flow
    question_sequence = [
        "contact_name", "email", "phone", "company_name", "role",
        "preferred_product", "moq", "preferred_communication", "custom_interest",
        "pricing_catalog_request", "sample_tasting_request", "early_access_interest", "customer_objections",
        # Company-specific fields that might not apply to all companies
        "interest_level", "business_type", "estimated_volume_kg_month", "primary_use_case",
        "gardening_experience_level", "garden_size", "farm_size", "current_supplier",
        "current_roaster", "current_ai_usage", "timeline_for_implementation", "specific_product_needs"
    ]
    
    # Find the next missing field based on priority sequence
    next_question = None
    for field in question_sequence:
        if field in missing_required or field in missing_optional:
            next_question = field
            break

    # --- Format Collected Data for Prompt ---
    collected_summary = "\n".join([f"- {key}: {value}" for key, value in collected_fields.items()]) if collected_fields else "None yet."

    # --- Get Available Product Varieties ---
    product_varieties = company_config.get('product_varieties', [])
    product_varieties_str = ", ".join(product_varieties) if product_varieties else "N/A"

    # --- Construct the Prompt ---
    prompt_lines = [
        f"You are a friendly and professional representative for '{company_config.get('display_name', 'Our Company')}' at the '{business_context.get('event_name', 'event')}'." ,
        f"Your primary goal is: {company_config.get('conversation_goal', 'Engage the user and gather information.')}",
        "\nCompany Background:",
        company_config.get('company_info', 'N/A'),
        "\nProducts/Services:",
        company_config.get('products_info', 'N/A'),
        f"\nProduct Varieties:",
        product_varieties_str,
        "\nConversation Context:",
        f"Information collected so far:\n{collected_summary}",
        f"Required information still needed: {', '.join(missing_required) if missing_required else 'None'}",
        f"Optional information still needed: {', '.join(missing_optional) if missing_optional else 'None'}",
        "\nYour Task:",
        "Review the recent conversation history.",
        "You should follow a specific conversation flow to gather information in the following order:",
        "1. Basic contact info - name, email, phone (offering to scan business card)",
        "2. Which product varieties they're interested in",
        "3. What minimum order quantity (MOQ) they're interested in",
        "4. Preferred communication method (WhatsApp/email/phone) for follow-up",
        "5. Interest in custom blends/solutions",
        "6. If they'd like a pricing catalog",
        "7. If they'd like to request sample tastings/demos",
        "8. Interest in early access to new products/blends",
        "9. Note any customer concerns or objections",
        "Based on this flow and missing information, ask *one* natural, conversational question at a time.",
        "Keep your responses concise and friendly.",
        "Do NOT ask multiple questions in a single turn.",
        "Do NOT list the fields you need. Ask questions naturally to get the information.",
        "Example: Instead of 'What is your email?', ask 'What's the best email address to reach you at?'",
    ]
    
    # Add specific guidance based on the next question to ask
    if next_question == "preferred_product":
        prompt_lines.append(f"Ask which product varieties they're interested in. Mention the available options: {product_varieties_str}")
    elif next_question == "moq":
        prompt_lines.append("Ask about their minimum order quantity (MOQ) requirements or expectations.")
    elif next_question == "preferred_communication":
        prompt_lines.append("Ask about their preferred method of communication for follow-up (WhatsApp, email, or phone).")
    elif next_question == "custom_interest":
        prompt_lines.append("Ask if they're interested in custom blends or customized solutions.")
    elif next_question == "pricing_catalog_request":
        prompt_lines.append("Ask if they would like to receive a pricing catalog.")
    elif next_question == "sample_tasting_request":
        prompt_lines.append("Ask if they would like to request samples for tasting/testing.")
    elif next_question == "early_access_interest":
        prompt_lines.append("Ask if they're interested in early access to new products or blends.")
    elif next_question == "customer_objections":
        prompt_lines.append("Ask if they have any concerns or questions about the products/services.")
    
    # Tool Usage Hint
    prompt_lines.append("If contact information (like name, email, phone, or company name) is missing, **offer to scan the user's business card** using the 'scan_business_card' tool as an efficient way to collect these details. Ask something like 'Do you happen to have a business card I could scan quickly?'")
    
    # Goal Completion Hint
    prompt_lines.append("If you believe you have collected all required information and fulfilled the conversation goal, you can politely steer towards ending the interaction (e.g., 'Thanks for the information! I'll make sure our team follows up. Enjoy the rest of the event!').")
    
    prompt_lines.append("Focus on the *next single step* in the conversation based on the established flow.")

    # Add specific instruction if a scan was just processed and confirmed/corrected
    if scan_just_processed:
        prompt_lines.append("\nImportant: You just presented scanned information, and the user likely just confirmed or corrected it in their last message. Briefly acknowledge their response (e.g., 'Got it, thanks!' or 'Okay, I've updated that.') and then proceed based on the *current* state of collected information by asking about the next item in the conversation flow. Do NOT repeat the scan confirmation details or ask 'Is this correct?' again.")

    final_prompt = "\n".join(prompt_lines)
    logger.debug(f"Generated Dynamic Prompt:\n{final_prompt}") # Log the full prompt at debug level
    return final_prompt

def agent_node(state: LeadState, config: RunnableConfig):
    """Prepares messages and invokes the LLM, handling different logic paths."""
    logger.info(f"--- Agent Node --- Current Stage (Informational): {state.get('current_stage', 'N/A')}")

    # Safety check for END state
    if state.get('current_stage') == END:
        logger.info("Agent node called in END state, skipping.")
        return {"error": None}

    current_history = state["conversation_history"]
    messages = []
    state_updates_after_prompt = {}
    response = None # Initialize response variable

    pending_image = state.get("pending_image_data")
    if pending_image:
        # --- Path 1: Pending Image (Business Card Scan) ---
        logger.info("Pending image data detected. Preparing minimal request for business card scan.")
        messages = [
            SystemMessage(content="An image of a business card has been provided. Use the 'scan_business_card' tool to extract the information. Do not engage in conversation, just call the tool.")
        ]
        logger.info("Prepared minimal messages for tool call (scan_business_card).")
        # No state_updates_after_prompt needed here usually

        # --- Invoke LLM (Image Path) ---
        try:
            logger.debug(f"Invoking LLM (image path) with {len(messages)} messages.")
            response = llm_with_tools.invoke(input=messages, config=config) # Assign to outer response
            logger.debug(f"LLM Response (image path): Type={type(response)}, Content={response.content}")
        except Exception as e:
            logger.error(f"Error invoking LLM (image path): {e}", exc_info=True)
            return {"error": f"LLM invocation failed (image path): {e}", "output": "Sorry, I encountered an error processing the image."}

    else:
        # --- Path 2: Regular Conversation Flow ---
        logger.info("No pending image data. Proceeding with standard conversation flow.")
        system_prompt = ""
        try:
            # Generate prompt and handle flags
            system_prompt = generate_dynamic_prompt(state)
            logger.info(f"--- Generated System Prompt Start ---\n{system_prompt}\n--- Generated System Prompt End ---") # Log the generated prompt for debugging context length
            if state.get("scan_just_processed"):
                state_updates_after_prompt["scan_just_processed"] = False # Prepare flag update

            # Prune history and prepare messages list
            MAX_HISTORY_MESSAGES = 10
            
            # First sanitize any existing business card tool calls in the history
            sanitized_history = []
            for msg in current_history:
                if isinstance(msg, AIMessage) and msg.tool_calls:
                    # Check if this message contains a business card scan tool call
                    if any(tc.get("name") == "scan_business_card" for tc in msg.tool_calls):
                        # Create sanitized tool calls
                        sanitized_tool_calls = []
                        for tc in msg.tool_calls:
                            sanitized_tc = tc.copy()
                            if tc.get("name") == "scan_business_card" and "args" in tc:
                                # Create sanitized args with placeholder
                                sanitized_args = tc.get("args", {}).copy()
                                sanitized_args["image_data"] = "<base64_encoded_image_data>"
                                sanitized_tc["args"] = sanitized_args
                            sanitized_tool_calls.append(sanitized_tc)
                        
                        # Create a new message with sanitized tool calls
                        sanitized_msg = AIMessage(
                            content=msg.content,
                            tool_calls=sanitized_tool_calls,
                            additional_kwargs=msg.additional_kwargs.copy() if hasattr(msg, "additional_kwargs") else {}
                        )
                        sanitized_history.append(sanitized_msg)
                    else:
                        sanitized_history.append(msg)
                else:
                    sanitized_history.append(msg)
            
            # Now prune the sanitized history
            pruned_history = sanitized_history[-MAX_HISTORY_MESSAGES:]
            messages = [SystemMessage(content=system_prompt)] + pruned_history
            logger.info(f"--- Inspecting Pruned History (Size: {len(pruned_history)}) ---")
            total_history_len = 0
            for i, msg in enumerate(pruned_history):
                msg_len = len(str(msg.content)) if msg.content else 0
                # Also check tool_calls size if it's an AIMessage, but limit the size
                tool_calls_len = 0
                if isinstance(msg, AIMessage) and msg.tool_calls:
                    # Just report if tool calls exist, but don't include the full content
                    # This prevents excessive logging for business card image data
                    if any(tc.get("name") == "scan_business_card" for tc in msg.tool_calls):
                        tool_calls_len = 100  # Placeholder size for reporting
                        logger.info(f"History Msg {i}: Type={type(msg).__name__}, Content Len={msg_len}, ToolCalls=scan_business_card (size limited), Total Approx Len={msg_len + 100}")
                    else:
                        # For non-business card tool calls, we can log the normal way
                        tool_calls_len = len(str(msg.tool_calls)) # Approx length of tool_calls structure
                        logger.info(f"History Msg {i}: Type={type(msg).__name__}, Content Len={msg_len}, ToolCalls Len={tool_calls_len}, Total Approx Len={msg_len + tool_calls_len}")
                else:
                    # For non-tool call messages
                    logger.info(f"History Msg {i}: Type={type(msg).__name__}, Content Len={msg_len}, ToolCalls Len=0, Total Approx Len={msg_len}")
                
                total_history_len += (msg_len + tool_calls_len)
            logger.info(f"--- Total Approx Length of Pruned History Content + ToolCalls: {total_history_len} ---")
            logger.info(f"Prepared messages with dynamic prompt, tools, and {len(pruned_history)} history messages.")

            # --- Invoke LLM (Regular Path) ---
            logger.debug(f"Invoking LLM (regular path) with {len(messages)} messages.")
            try:
                response = llm_with_tools.invoke(input=messages, config=config) # Assign to outer response
                logger.debug(f"LLM Response (regular path): Type={type(response)}, Content={response.content}")
            except Exception as token_error:
                # Check if it's a token limit error
                error_str = str(token_error)
                if "maximum context length" in error_str and "context_length_exceeded" in error_str:
                    logger.warning("Token limit exceeded. Attempting emergency history pruning and retry...")
                    
                    # Emergency fallback: Drastically reduce history and retry
                    # Keep only the last 3 messages and sanitize them
                    emergency_history = []
                    if len(pruned_history) > 0:
                        # Add last human message if available
                        human_msgs = [msg for msg in pruned_history if isinstance(msg, HumanMessage)]
                        if human_msgs:
                            emergency_history.append(human_msgs[-1])
                    
                    # Create completely fresh messages list with only system message and maybe last human message
                    emergency_messages = [SystemMessage(content=system_prompt)] + emergency_history
                    logger.info(f"Emergency retry with {len(emergency_history)} history messages")
                    
                    # Try again with drastically reduced history
                    response = llm_with_tools.invoke(input=emergency_messages, config=config)
                    logger.info("Successfully recovered from token limit error with emergency history pruning")
                else:
                    # Re-raise if it's not a token limit error
                    raise

        except Exception as e:
            # This catches errors in prompt generation, history pruning, OR LLM invocation
            logger.error(f"Error during regular conversation flow (prompt/prune/invoke): {e}", exc_info=True)
            return {"error": f"Failed during regular flow: {e}", "output": "Sorry, an internal error occurred preparing my response."}

    # --- Process Response (Common to both paths if LLM succeeded) ---
    if response is None:
        # This should ideally not be reached if exceptions are caught correctly, but acts as a safeguard
        logger.error("Critical: LLM response is None after invoke blocks. Check logic paths.")
        return {"error": "Internal state error: LLM response missing.", "output": "Sorry, an unexpected internal error occurred."}

    tool_calls = parse_tool_calls(response)
    logger.debug(f"Parsed tool calls: {tool_calls}")

    updated_history = current_history + ([response] if isinstance(response, BaseMessage) else [])

    final_updates = {
        "conversation_history": updated_history,
        "output": response.content if isinstance(response, BaseMessage) else str(response),
        "error": None
    }
    final_updates.update(state_updates_after_prompt) # Merge flag updates

    logger.info("Agent node finished successfully.")
    return final_updates


# --- [REMOVED] Old stage-based logic functions ---
# process_profile_update_node, determine_next_stage_node, entry_router, check_for_pause
# These are replaced by the dynamic prompt and simpler routing logic below.


# --- Graph Definition ---

# Create the ToolNode with the available tools
# The ToolNode itself, remains unchanged
_tool_node = ToolNode(agent_tools)

# --- Wrapper Function for Tool Execution ---
def execute_tools_node(state: LeadState) -> Dict[str, List[BaseMessage]]:
    """
    Extracts messages, executes tools, and appends ToolMessages back to history.
    """
    logger.info("--- Tool Execution Node ---")
    messages = state.get("conversation_history", [])
    last_message = messages[-1] if messages else None

    # --- Inject Pending Image Data into Tool Call Args ---
    pending_image = state.get("pending_image_data")
    if pending_image and isinstance(last_message, AIMessage) and last_message.tool_calls:
        modified_tool_calls = []
        image_data_injected = False
        original_tool_calls = []
        for tool_call in last_message.tool_calls:
            if tool_call.get("name") == "scan_business_card":
                # Store original tool call for history
                original_tool_call = tool_call.copy()
                # Create placeholder args showing the image was provided but not storing it
                placeholder_args = {"image_data": "<base64_encoded_image_data>"}  
                original_tool_call["args"] = placeholder_args
                original_tool_calls.append(original_tool_call)
                
                # Create a mutable copy of the args dictionary for execution
                modified_args = tool_call.get("args", {}).copy()
                # Inject the actual image data
                modified_args["image_data"] = pending_image
                # Create a new tool call dict with modified args
                modified_tool_call = tool_call.copy()
                modified_tool_call["args"] = modified_args
                modified_tool_calls.append(modified_tool_call)
                image_data_injected = True
                logger.info(f"Injected pending image data into tool call ID {tool_call.get('id')} for scan_business_card.")
            else:
                modified_tool_calls.append(tool_call) # Keep other tool calls as is
                original_tool_calls.append(tool_call.copy()) # Also keep for history

        if image_data_injected:
            # Create a new AIMessage with sanitized tool calls for conversation history
            history_message = AIMessage(
                content=last_message.content,
                tool_calls=original_tool_calls,
                additional_kwargs=last_message.additional_kwargs.copy() if hasattr(last_message, "additional_kwargs") else {}
            )
            
            # Replace the last message in history with our sanitized version
            # This prevents the full image data from being stored in conversation history
            messages_copy = messages.copy()
            if messages_copy:
                messages_copy[-1] = history_message
                state["conversation_history"] = messages_copy
            
            # Update the tool_calls on the last_message for execution
            last_message.tool_calls = modified_tool_calls
            
            # Clear the pending image data from the state after injecting it
            state["pending_image_data"] = None
            logger.debug("Cleared pending_image_data from state and sanitized conversation history.")
    # --- End Image Data Injection ---


    if not last_message or not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        logger.warning("Tool execution node called without tool calls in the last message (or after potential modification).")
        return {} # No changes to state if no valid tool calls found

    # Invoke the ToolNode with the potentially modified messages list
    try:
        tool_messages = _tool_node.invoke(messages) # Pass the list directly
    except Exception as e:
        logger.error(f"Error executing tools: {e}", exc_info=True)
        # Handle tool execution error, maybe add an error message to history or state
        # For now, just return empty to avoid breaking state structure
        # TODO: Add proper error handling/message generation here
        return {"error": f"Tool execution failed: {e}"}


    # LangGraph ToolNode returns a ToolMessage or list of ToolMessages
    if not isinstance(tool_messages, list):
        tool_messages = [tool_messages] # Ensure it's a list

    logger.debug(f"Tool execution resulted in messages: {tool_messages}")

    # Append the tool results to the conversation history
    logger.info(f"--- Tool Execution Result Messages Start ---") # Changed to INFO
    for tm in tool_messages:
        logger.info(f"ToolMessage Content (Type: {type(tm.content)}, Length: {len(str(tm.content))}): {str(tm.content)[:500]}...") # Changed to INFO
    logger.info(f"--- Tool Execution Result Messages End ---") # Changed to INFO
    updated_history = messages + tool_messages

    # Return the updated history
    # Also clear any previous error if tool execution succeeded
    return {"conversation_history": updated_history, "error": None}

# --- New Node for Processing Scan Results ---
def process_scan_result_node(state: LeadState) -> Dict[str, Any]:
    """
    Processes the result from the business card scan tool.
    Updates state with extracted data and prepares a confirmation message or error message.
    """
    logger.info("--- Process Scan Result Node ---")
    messages = state.get("conversation_history", [])
    last_message = messages[-1] if messages else None

    output_message = "An unexpected error occurred after scanning." # Default error
    updated_collected_fields = state.get("collected_fields", {}).copy()
    scan_successful = False
    tool_call_id_for_message = None # To link AIMessage back if needed

    # Check if the last message is a ToolMessage
    if isinstance(last_message, ToolMessage):
        tool_call_id_for_message = last_message.tool_call_id # Get ID to potentially link AI response
        try:
            # Infer tool name from additional_kwargs if available (depends on ToolNode implementation)
            # tool_name = last_message.additional_kwargs.get("name", "") # Less reliable
            # More reliably, check if the content looks like our scan tool output
            is_scan_result = False
            scan_result = {}
            try:
                content_dict = json.loads(last_message.content)
                if isinstance(content_dict, dict) and "status" in content_dict:
                     # Assume any ToolMessage with a 'status' key is from our scan tool for now
                     is_scan_result = True
                     scan_result = content_dict
            except (json.JSONDecodeError, TypeError):
                 is_scan_result = False

            if not is_scan_result:
                 # If it's not a scan result, maybe it's another tool? Log and pass through?
                 # For now, assume only scan tool leads here. Raise error if not scan result.
                 logger.warning(f"process_scan_result node received non-scan ToolMessage content: {last_message.content[:100]}...")
                 # Let's return an error state or a generic message
                 output_message = "Received an unexpected tool result. Please continue."
                 # Keep collected_fields unchanged if it wasn't a successful scan
                 updated_collected_fields = state.get("collected_fields", {})

            else: # It is a scan result, process it
                logger.debug(f"Processing ToolMessage content: {last_message.content}")
                status = scan_result.get("status")

                if status == "success":
                    logger.info("Scan successful. Extracting details.")
                    scan_successful = True
                    # Filter out status and None values, keep empty strings if present
                    extracted_data = {k: v for k, v in scan_result.items() if k != "status" and v is not None}

                    # Update collected fields, preferring new non-empty values
                    for key, value in extracted_data.items():
                         if value: # Only update if the new value is not empty/None
                              # Map standard business card fields to company-required fields
                              if key == "full_name":
                                  updated_collected_fields["contact_name"] = value
                              elif key == "company":
                                  updated_collected_fields["company_name"] = value
                              elif key == "title" or key == "position":
                                  updated_collected_fields["role"] = value
                              else:
                                  updated_collected_fields[key] = value

                    # Format confirmation message
                    confirmation_lines = ["Okay, I scanned the card. Here's what I found:"]
                    display_data = {k: v for k, v in extracted_data.items() if v} # Only show non-empty fields
                    if not display_data:
                         confirmation_lines.append("I couldn't extract any details from the card.")
                    else:
                        for key, value in display_data.items():
                            # Make keys more readable if needed (e.g., 'full_name' -> 'Full Name')
                            readable_key = key.replace('_', ' ').title()
                            confirmation_lines.append(f"- {readable_key}: {value}")
                    confirmation_lines.append("\nIs this information correct?")
                    output_message = "\n".join(confirmation_lines)

                elif status == "error":
                    error_detail = scan_result.get("message", "Unknown scan error.")
                    logger.warning(f"Scan tool returned an error: {error_detail}")
                    output_message = f"Sorry, I encountered an issue scanning the card: {error_detail}. Could you please provide the details manually?"
                elif status == "disabled":
                     logger.warning("Scan tool is disabled.")
                     output_message = "Sorry, the business card scanning feature is currently unavailable. Could you please provide the details manually?"
                else:
                     logger.error(f"Scan tool returned unexpected status: {status}")
                     output_message = "Sorry, there was an unexpected issue after trying to scan the card. Let's try entering the details manually."

        except json.JSONDecodeError:
            logger.error(f"Failed to decode JSON from ToolMessage: {last_message.content}", exc_info=True)
            output_message = "Sorry, I had trouble reading the scan result. Could you provide the details manually?"
        except Exception as e:
            logger.error(f"Error processing scan result: {e}", exc_info=True)
            output_message = f"Sorry, an unexpected error occurred while processing the scan: {e}. Let's try manually."
    else:
        logger.warning("Process Scan Result node called without a preceding ToolMessage.")
        # This case shouldn't happen with correct routing, but handle defensively.
        output_message = "It seems there was a mix-up. Could you please provide your contact details?"


    # --- Prepare State Updates ---
    updates = {
        "output": output_message, # This message content will be sent to the user by the API layer
        "collected_fields": updated_collected_fields,
        "error": None, # Clear any previous errors
        "scan_just_processed": True # Set the flag as scan result was just processed
    }
    # We might want a specific stage for confirmation later, e.g., "confirming_scan"
    # updates["current_stage"] = "confirming_scan" if scan_successful else state.get("current_stage")


    # --- Add AI Message to History ---
    # The AI needs to "say" the confirmation or error message.
    # We add this AIMessage to the history here. The API layer should send state["output"].
    ai_confirm_message = AIMessage(content=output_message)
    # If we have the tool_call_id, we can link this AIMessage, although it's not strictly necessary here
    # as it's a response *to* the tool result, not the tool result itself.
    # ai_confirm_message.tool_calls = None # Ensure no phantom tool calls

    current_history = state.get("conversation_history", [])
    # Avoid adding duplicate messages if node reruns somehow
    if not current_history or current_history[-1].content != ai_confirm_message.content:
         # Check if the last message was the ToolMessage we just processed
         if len(current_history) > 0 and isinstance(current_history[-1], ToolMessage) and current_history[-1].tool_call_id == tool_call_id_for_message:
              current_history.append(ai_confirm_message)
         # Or if the history is empty or last message is different
         elif len(current_history) == 0 or not isinstance(current_history[-1], AIMessage) or current_history[-1].content != ai_confirm_message.content:
              current_history.append(ai_confirm_message)

    updates["conversation_history"] = current_history # Ensure history update is returned

    logger.debug(f"Process Scan Result node finished. Output: '{output_message[:100]}...'")
    return updates

# --- New Conditional Logic Function ---
def route_after_agent(state: LeadState) -> str:
    """
    Determines the next step after the agent node runs.
    - If the last message has tool calls, route to the 'tools' node.
    - Otherwise, end the current graph invocation (wait for user input).
    """
    last_message = state.get("conversation_history", [])[-1] if state.get("conversation_history") else None

    # Check if the last message is an AIMessage and has tool_calls
    if isinstance(last_message, AIMessage) and getattr(last_message, 'tool_calls', None):
         # Ensure tool_calls is not empty
         if last_message.tool_calls:
             logger.info(f"Routing: Agent -> Tools (Tool calls found: {last_message.tool_calls})")
             return "tools"
         else:
             # Handle cases where tool_calls attribute exists but is empty (shouldn't happen with valid LLM output)
             logger.warning("Routing: Agent -> END (Empty tool_calls attribute found in last message)")
             return END
    else:
         # No tool calls detected in the last message
         logger.info("Routing: Agent -> END (No tool calls detected in last message, waiting for user input)")
         return END

# --- Build Graph Function (Refactored) ---
def build_graph():
    """Builds the simplified LangGraph StateGraph for dynamic conversation flow."""
    graph_builder = StateGraph(LeadState)

    # Add nodes
    graph_builder.add_node("agent", agent_node)
    graph_builder.add_node("tools", execute_tools_node) # Executes the tool(s)
    graph_builder.add_node("process_scan_result", process_scan_result_node) # Processes scan result

    # Set entry point directly to the agent
    graph_builder.set_entry_point("agent")

    # Define edges
    # After agent runs, decide whether to call tools or end the turn
    graph_builder.add_conditional_edges(
        "agent",
        route_after_agent, # Use the new routing function
        {
            "tools": "tools", # If tool calls exist, go to tools node
            END: END          # If no tool calls, end the graph invocation
        }
    )

    # After tools run, go to process the result (specifically for scan)
    # TODO: This needs refinement if other tools are added that don't need this processing step.
    graph_builder.add_edge("tools", "process_scan_result")

    # After processing the scan result, the flow should wait for user input (confirmation/correction).
    # The process_scan_result node adds the confirmation/error AIMessage to history.
    # Ending the turn here allows the API layer to send that message.
    # The next user input will trigger the graph again starting at 'agent'.
    graph_builder.add_edge("process_scan_result", END)

    # Compile the graph
    graph = graph_builder.compile()
    logger.info("Dynamic conversation graph compiled.")
    return graph


# --- Initialization Function ---
# (initialize_session remains largely the same, state fields added in main.py)
def initialize_session(session_id: str, business_context: Dict) -> LeadState:
    """
    Initializes the state for a new session.
    Note: company_config and collected_fields are added in main.py after this call.
    """
    # Initialize with default values for all keys defined in LeadState
    return LeadState(
        input=None,
        output=None,
        conversation_history=[],
        current_stage="greeting", # Initial stage, will become less rigid
        stage_history=[],
        lead_profile={},
        business_context=business_context,
        session_id=session_id,
        tool_calls=None,
        error=None,
        company_config=None,
        collected_fields={},
        pending_image_data=None, # Initialize pending image data
        scan_just_processed=False, # Initialize the new flag
    )


# --- Global Graph Instance ---
# Ensure the graph is built when the module is imported
trade_show_agent_graph = build_graph()


# --- Main Execution Block (for testing if needed) ---
if __name__ == '__main__':
    # Example usage for testing the graph directly (optional)
    logging.basicConfig(level=logging.DEBUG) # Set to DEBUG for detailed graph logs
    logger.info("--- Running langraph_agent.py directly for testing ---")

    # Example: Initialize state manually for testing
    test_session_id = "test_session_001"
    test_business_context = {
        "company_name": "Test Company",
        "event_name": "Test Event",
        "company_info": "We do testing.",
        "products_info": "Test products."
    }
    # Load a specific company config for testing
    from .config import load_company_configs
    test_company_configs = load_company_configs()
    test_config_key = "abc_roosters" # Choose a config to test
    test_company_config = test_company_configs.get(test_config_key, test_company_configs['default'])

    initial_state = initialize_session(test_session_id, test_business_context)
    initial_state["company_config"] = test_company_config
    initial_state["collected_fields"] = {} # Start with no fields collected

    # Example: Simulate an initial user message
    initial_state["conversation_history"].append(HumanMessage(content="Hello there!"))
    initial_state["input"] = "Hello there!"

    # Configuration for invoking the graph
    test_config = {"configurable": {"session_id": test_session_id}, "recursion_limit": 10}

    logger.info(f"\n--- Invoking graph with initial state for {test_config_key} ---")
    try:
        # Use stream or invoke - stream shows intermediate steps
        # for step in trade_show_agent_graph.stream(initial_state, config=test_config):
        #     print(f"Step Output:\n{json.dumps(step, indent=2, default=str)}\n---")

        # Or just invoke to get the final state after one turn
        final_state = trade_show_agent_graph.invoke(initial_state, config=test_config)
        logger.info("\n--- Final State after invoke ---")
        # Use json.dumps for cleaner printing of complex state
        logger.info(json.dumps(final_state, indent=2, default=str))

    except Exception as e:
        logger.error(f"Error during graph test execution: {e}", exc_info=True)
