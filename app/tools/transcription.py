import os
import json
import logging
from typing import List, Dict, Any, Optional

# Get a logger for this module
logger = logging.getLogger(__name__)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.output_parsers import OutputFixingParser # Corrected import path
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI # Keep as an alternative
from langchain.tools import tool
from langchain_core.callbacks import CallbackManagerForToolRun
from pydantic import ValidationError, BaseModel
from langchain_core.language_models.chat_models import BaseChatModel # For type hinting

# Assuming models are in the parent directory structure
from ..models.lead import LeadProfile
from ..models.conversation import ConversationMessage
from ..config import is_feature_enabled, get_llm_config # Import config checker

# Check LLM API keys only if feature is intended to be enabled
ANALYSIS_ENABLED = is_feature_enabled('conversation_analysis.enabled')
llm_config = get_llm_config()
primary_provider = llm_config.get("primary_provider", "anthropic")
anthropic_key_present = bool(os.getenv("ANTHROPIC_API_KEY"))
openai_key_present = bool(os.getenv("OPENAI_API_KEY"))
llm_available = anthropic_key_present or openai_key_present

if ANALYSIS_ENABLED and not llm_available:
    logger.warning("Conversation analysis feature is enabled but no LLM API key (ANTHROPIC_API_KEY or OPENAI_API_KEY) is set. Analysis will fail.")
elif not ANALYSIS_ENABLED:
    logger.info("TranscriptionTool (Conversation Analysis) is disabled by configuration.")


class TranscriptionTool:
    """Tool for analyzing conversations using an LLM and extracting structured lead information. Configurable via config.yaml."""

    def __init__(self):
        """Initializes the tool based on configuration."""
        self.enabled = ANALYSIS_ENABLED and llm_available
        self.llm = None
        self.output_parser = PydanticOutputParser(pydantic_object=LeadProfile)
        self.output_fixing_parser = None
        self.init_error = None

        if self.enabled:
            model_name = None
            fallback_model = None
            llm_initialized = False

            # Determine which LLM to try first based on config
            if primary_provider == "anthropic" and anthropic_key_present:
                model_name = llm_config.get("anthropic_model")
                try:
                    self.llm = ChatAnthropic(model=model_name, temperature=llm_config.get("temperature", 0))
                    logger.info(f"TranscriptionTool using primary LLM: Anthropic '{model_name}'")
                    llm_initialized = True
                except Exception as e:
                    self.init_error = f"Failed to initialize Anthropic model '{model_name}': {e}"
                    logger.warning(f"{self.init_error}", exc_info=True)

            # Try OpenAI if Anthropic failed or wasn't primary/available
            if not llm_initialized and openai_key_present:
                fallback_model = llm_config.get("openai_model")
                try:
                    self.llm = ChatOpenAI(model=fallback_model, temperature=llm_config.get("temperature", 0))
                    logger.info(f"TranscriptionTool using fallback LLM: OpenAI '{fallback_model}'")
                    llm_initialized = True
                except Exception as e:
                    # Append error if Anthropic also failed
                    error_msg = f"Failed to initialize OpenAI model '{fallback_model}': {e}"
                    self.init_error = f"{self.init_error}. {error_msg}" if self.init_error else error_msg
                    logger.warning(f"{error_msg}", exc_info=True)

            # If still no LLM, disable the tool
            if not llm_initialized:
                self.init_error = self.init_error or "No suitable LLM API key found or model initialization failed."
                logger.error(f"{self.init_error}")
                self.enabled = False
            else:
                # Initialize OutputFixingParser with the chosen LLM
                try:
                    self.output_fixing_parser = OutputFixingParser.from_llm(parser=self.output_parser, llm=self.llm)
                except Exception as e:
                     # Log this error but don't necessarily disable the tool, maybe basic parsing can still work
                     logger.warning(f"Failed to initialize OutputFixingParser: {e}", exc_info=True)
                     self.output_fixing_parser = None # Fallback to basic parser

        elif ANALYSIS_ENABLED and not llm_available:
             self.init_error = "No LLM API key (ANTHROPIC_API_KEY or OPENAI_API_KEY) provided."

# --- Standalone Helper Functions ---

def _format_conversation_func(conversation_history: List[Dict[str, Any]]) -> str:
    """Formats conversation history list into a readable string for the LLM."""
    if not conversation_history:
        return ""

    formatted_lines = []
    for msg in conversation_history:
        role = msg.get("role", "unknown").capitalize()
        content = msg.get("content", "")
        # Standardize roles for clarity in the prompt
        if role == "Human":
            speaker = "Visitor"
        elif role == "Assistant": # Changed from AIMessage type check to role check
            speaker = "Assistant"
        elif role == "System":
             speaker = "System"
        elif role == "Tool":
             speaker = f"Tool ({msg.get('tool_call_id', '')})" # Add tool call ID if available
             content = f"Result: {content}" # Clarify tool result
        else:
            speaker = role # Keep other roles as is

        formatted_lines.append(f"{speaker}: {content}")

    return "\n".join(formatted_lines)


def _analyze_conversation_func(
    conversation_history_json: str,
    llm: BaseChatModel,
    output_parser: PydanticOutputParser,
    output_fixing_parser: Optional[OutputFixingParser],
    enabled: bool,
    init_error: Optional[str]
) -> str:
    """
    Standalone function to analyze conversation history and extract lead info.
    """
    if not enabled:
         return json.dumps({"status": "disabled", "message": f"Conversation analysis feature is disabled or not configured correctly. {init_error or ''}".strip()})

    try:
        # Use basic parser if fixing parser failed to init
        current_parser = output_fixing_parser or output_parser

        # Parse the conversation history
        try:
            conversation_history = json.loads(conversation_history_json)
        except json.JSONDecodeError:
            return json.dumps({"status": "error", "message": "Invalid JSON format for conversation history."})

        # Format the conversation for the LLM
        formatted_conversation = _format_conversation_func(conversation_history) # Call helper
        if not formatted_conversation:
             return json.dumps({"status": "error", "message": "Conversation history is empty or invalid."})

        # Create the prompt
        prompt_template = """
You are an expert conversation analyzer specializing in trade show interactions.
Your task is to meticulously extract detailed information about a potential lead from the provided conversation transcript between an AI assistant ('Assistant') and a visitor ('Visitor' or 'Human').

Analyze the conversation and extract the following details, filling in as much information as possible based *only* on the conversation content:
1.  **Basic Contact Info:** Full name, company name, job title/role, email, phone number, location/city, website.
2.  **Product Interest:** Specific products, services, or categories the lead expressed interest in.
3.  **Order Quantity (MOQ):** Any mention of desired order size, minimum order quantity, or trial batch size.
4.  **Concerns/Pain Points:** Any objections, questions about pricing, quality, delivery, or challenges they face.
5.  **Interest Level:** Assess their buying intent based on their engagement, questions, and statements (e.g., Low, Medium, Medium-High, High).
6.  **Preferred Contact:** How they prefer to be contacted for follow-ups (e.g., WhatsApp, Email).
7.  **Follow-up Actions:** Any specific actions requested or agreed upon (e.g., send samples, schedule demo, provide pricing, custom blend discussion).
8.  **Event Context:** Note the event name if mentioned.

{format_instructions}

Conversation Transcript:
----------------------
{conversation}
----------------------

Based *only* on the transcript above, provide the extracted information in the requested JSON format. If a piece of information is not mentioned, omit the field or leave it as null/empty list as appropriate according to the format instructions. Do not invent information not present in the text.
"""
        prompt = ChatPromptTemplate.from_template(
            template=prompt_template,
            partial_variables={"format_instructions": output_parser.get_format_instructions()}
        )

        # Decide which parser to use
        if output_fixing_parser:
             chain = prompt | llm | output_fixing_parser
             logger.info("Using OutputFixingParser for conversation analysis.")
        else:
             chain = prompt | llm | output_parser
             logger.info("Using basic PydanticOutputParser for conversation analysis.")


        # Invoke the chain
        response = chain.invoke({"conversation": formatted_conversation})

        # Return the Pydantic model as JSON
        # Ensure response is a Pydantic model before dumping
        if isinstance(response, BaseModel):
             return response.model_dump_json()
        else:
             # Handle cases where the parser might return a dict or string on failure
             logger.warning(f"Parser did not return a Pydantic model. Output: {response}")
             # Attempt to serialize if dict, otherwise return error
             if isinstance(response, dict):
                  try:
                       # Try creating the model instance manually for validation
                       LeadProfile.model_validate(response)
                       return json.dumps(response)
                  except ValidationError as ve:
                       logger.error(f"LLM output failed validation even after manual check: {ve}", exc_info=True)
                       return json.dumps({"status": "error", "message": f"LLM output failed validation: {ve}"})
             else:
                  logger.error(f"LLM output could not be structured. Raw output: {response}")
                  return json.dumps({"status": "error", "message": "LLM output could not be structured."})


    except ConnectionError as ce:
         logger.error(f"LLM connection error during analysis: {ce}", exc_info=True)
         return json.dumps({"status": "error", "message": str(ce)})
    except ValidationError as ve:
         # This might happen if the LLM output doesn't match the Pydantic model *after* fixing attempt
         logger.error(f"Failed to structure LLM output after parsing/fixing: {ve}", exc_info=True)
         return json.dumps({"status": "error", "message": f"Failed to structure LLM output: {ve}"})
    except Exception as e:
        logger.error(f"Unexpected error analyzing conversation: {e}", exc_info=True)
        return json.dumps({"status": "error", "message": f"Error analyzing conversation: {str(e)}"})


# Create an instance of the configuration/state holder class
_transcription_tool_instance = TranscriptionTool()

# Define the actual LangChain tool using the standalone function and the instance's state
@tool("analyze_conversation")
def analyze_conversation_tool(conversation_history_json: str) -> str:
    """
    Analyzes the conversation history JSON string to extract structured lead information.
    Input must be a JSON string representing a list of message dictionaries.
    Returns a JSON string containing the extracted LeadProfile data or an error message.
    """
    return _analyze_conversation_func(
        conversation_history_json=conversation_history_json,
        llm=_transcription_tool_instance._get_active_llm(), # Get LLM from instance
        output_parser=_transcription_tool_instance.output_parser,
        output_fixing_parser=_transcription_tool_instance.output_fixing_parser,
        enabled=_transcription_tool_instance.enabled,
        init_error=_transcription_tool_instance.init_error
    )

# The object to import should be the decorated function itself
# (LangChain's @tool decorator turns the function into a Runnable/Tool object)
# We will import 'analyze_conversation_tool' in main.py now.
