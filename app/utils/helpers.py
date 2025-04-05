import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

# Get a logger for this module
logger = logging.getLogger(__name__)

# Assuming models are defined elsewhere, e.g., app.models.lead
# from ..models.lead import LeadProfile # Import if needed for type hinting

def format_lead_info_summary(lead_info: Optional[Dict[str, Any]]) -> str:
    """
    Formats lead information dictionary into a readable summary string
    for inclusion in LLM prompts.
    """
    if not lead_info:
        return "No information collected yet."

    summary_parts = []
    # Prioritize key fields
    if name := lead_info.get("full_name"):
        summary_parts.append(f"- Name: {name}")
    if company := lead_info.get("company"):
        summary_parts.append(f"- Company: {company}")
    if role := lead_info.get("role"):
        summary_parts.append(f"- Role: {role}")
    if location := lead_info.get("location"):
        summary_parts.append(f"- Location: {location}")
    if products := lead_info.get("products_of_interest"):
        summary_parts.append(f"- Interested In: {', '.join(products)}")
    if moq := lead_info.get("moq"):
        summary_parts.append(f"- MOQ/Quantity: {moq}")
    if concerns := lead_info.get("concerns"):
        summary_parts.append(f"- Concerns: {', '.join(concerns)}")
    if contact_pref := lead_info.get("preferred_contact"):
        summary_parts.append(f"- Preferred Contact: {contact_pref}")
    if interest := lead_info.get("interest_level"):
        summary_parts.append(f"- Assessed Interest: {interest}")
    # Add other fields if they become important for context
    if email := lead_info.get("email"):
         summary_parts.append(f"- Email: {email}")
    if phone := lead_info.get("phone"):
         summary_parts.append(f"- Phone: {phone}")


    if not summary_parts:
        return "No specific details collected yet."

    return "Visitor Information Summary:\n" + "\n".join(summary_parts)

def format_follow_up_summary(follow_up_actions: Optional[List[str]]) -> str:
    """
    Formats a list of follow-up action strings into a readable summary.
    """
    if not follow_up_actions:
        return "No specific follow-up actions agreed upon yet."

    summary_parts = [f"- {action}" for action in follow_up_actions]

    return "Agreed Follow-up Actions:\n" + "\n".join(summary_parts)

def format_conversation_history(conversation_history: List[Dict[str, Any]], max_messages: Optional[int] = None) -> str:
    """Formats conversation history list into a readable string for LLM analysis or context."""
    if not conversation_history:
        return "No conversation history."

    formatted_lines = []
    messages_to_format = conversation_history[-max_messages:] if max_messages else conversation_history

    for msg in messages_to_format:
        role = msg.get("role", "unknown").capitalize()
        content = msg.get("content", "")
        timestamp_str = msg.get("timestamp", "")

        # Try to parse timestamp for relative time, fallback to string
        time_display = ""
        if timestamp_str:
            try:
                ts = datetime.fromisoformat(timestamp_str)
                # Basic relative time could be added here if needed
                time_display = f" ({ts.strftime('%H:%M')})" # Simple time format
            except (ValueError, TypeError):
                time_display = f" ({timestamp_str})" # Fallback

        # Standardize roles for clarity
        if role == "Human":
            speaker = "Visitor"
        elif role == "Assistant":
            speaker = "Assistant"
        else:
            speaker = role # Keep system/tool roles as is

        formatted_lines.append(f"{speaker}{time_display}: {content}")

    return "\n".join(formatted_lines)

def parse_tool_calls(message: Any) -> Optional[List[Dict[str, Any]]]:
    """
    Safely extracts tool calls from a LangChain message object.

    Args:
        message: The message object from the LLM (e.g., AIMessage).

    Returns:
        A list of tool call dictionaries if present, otherwise None.
        Each dictionary typically contains 'name', 'args', and 'id'.
    """
    if hasattr(message, 'tool_calls') and message.tool_calls:
        # Ensure it's a list of dicts as expected by LangGraph ToolNode
        if isinstance(message.tool_calls, list) and all(isinstance(tc, dict) for tc in message.tool_calls):
             # Check for required keys (name, args, id) - LangChain AIMessage usually provides these
             # Example structure: [{'name': 'tool_name', 'args': {'arg1': 'val1'}, 'id': 'call_abc123'}]
             # No need for deep validation here, assume LangChain structure is correct if attribute exists.
            return message.tool_calls
        else:
            logger.warning(f"'tool_calls' attribute found but not in expected format (list of dicts): {message.tool_calls}")
            return None
    return None

# Add other helper functions as needed, e.g., for data cleaning, validation, etc.
