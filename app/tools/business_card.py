import base64
import json
import os
import logging
from typing import Optional, Dict, Any, List, Type

# Get a logger for this module
logger = logging.getLogger(__name__)

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
# Remove @tool decorator import, use BaseTool inheritance instead
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, ValidationError

# Assuming models are in the parent directory structure
from ..models.lead import BusinessCardInfo
from ..config import is_feature_enabled, get_llm_config # Import config checker
from dotenv import load_dotenv

path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(path)

# Check credentials only if feature is intended to be enabled
OCR_ENABLED = is_feature_enabled('ocr.enabled')
OPENAI_KEY_PRESENT = bool(os.getenv("OPENAI_API_KEY"))

if OCR_ENABLED and not OPENAI_KEY_PRESENT:
    logger.warning("OCR feature is enabled but OPENAI_API_KEY environment variable not set. OpenAI Vision API calls will fail.")
elif not OCR_ENABLED:
    logger.info("BusinessCardTool (OCR via OpenAI Vision) is disabled by configuration.")

# --- Define Input Schema ---
class BusinessCardInput(BaseModel):
    """Input schema for the BusinessCardTool."""
    image_data: str = Field(description="Base64 encoded string of the business card image (JPEG or PNG recommended).")

# --- Tool Implementation ---
class BusinessCardTool(BaseTool):
    """
    Tool for scanning business cards using OpenAI's vision capabilities (gpt-4o)
    and extracting structured contact information. Configurable via config.yaml.
    """
    name: str = "scan_business_card"
    description: str = (
        "Scans a business card image (base64 encoded) using OpenAI's gpt-4o model "
        "and attempts to extract structured contact information based on the BusinessCardInfo model. "
        "Returns a JSON string representing a BusinessCardInfo object with extracted details, "
        "or a JSON string indicating the feature is disabled or an error occurred."
    )
    args_schema: Type[BaseModel] = BusinessCardInput

    # Declare fields at class level for Pydantic validation
    # Use default_factory for mutable types like the parser
    # Mark client as Optional and default to None
    enabled: bool = Field(default=False)
    client: Optional[ChatOpenAI] = Field(default=None)
    output_parser: PydanticOutputParser = Field(default_factory=lambda: PydanticOutputParser(pydantic_object=BusinessCardInfo))
    init_error: Optional[str] = Field(default=None)

    def __init__(self, **kwargs):
        # Initialize BaseTool first - this triggers Pydantic validation
        super().__init__(**kwargs)

        # Now, update the fields based on runtime checks
        # Pydantic fields are already initialized with defaults by super().__init__
        _enabled_runtime = OCR_ENABLED and OPENAI_KEY_PRESENT
        _init_error_runtime = None
        _client_runtime = None

        if _enabled_runtime:
            try:
                _client_runtime = ChatOpenAI(model="gpt-4o", temperature=0)
                logger.info("OpenAI client (gpt-4o) initialized successfully for BusinessCardTool instance.")
            except Exception as e:
                _init_error_runtime = f"Error initializing OpenAI client: {e}. Ensure API key is valid."
                logger.warning(f"{_init_error_runtime}", exc_info=True)
                _enabled_runtime = False # Disable if client fails to init
        elif OCR_ENABLED and not OPENAI_KEY_PRESENT:
             _init_error_runtime = "OPENAI_API_KEY not found in environment variables."
             _enabled_runtime = False # Ensure disabled if key missing

        # Update the instance attributes (which are now validated Pydantic fields)
        # Use object.__setattr__ to bypass Pydantic's validation during this update,
        # as the values are determined post-initial validation.
        object.__setattr__(self, 'enabled', _enabled_runtime)
        object.__setattr__(self, 'client', _client_runtime)
        object.__setattr__(self, 'init_error', _init_error_runtime)
        # output_parser is handled by default_factory, no need to update here unless logic changes

    def _run(
        self, image_data: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        # Now check instance attributes
        if not self.enabled:
            return json.dumps({"status": "disabled", "message": f"OCR feature is disabled or OpenAI key is missing. {self.init_error or ''}".strip()})
        if not self.client:
            return json.dumps({"status": "error", "message": f"OpenAI client not available. {self.init_error or 'Client not initialized.'}".strip()})

        try:
            # --- Input Validation ---
            # Check for None, empty, placeholder, or invalid format (must be data URL or external URL)
            is_placeholder = image_data == "[Business Card Image Data]"
            is_data_url = isinstance(image_data, str) and image_data.startswith("data:image/")
            # Add check for http/https URLs if those are ever expected (currently assuming data URLs)
            # is_http_url = isinstance(image_data, str) and (image_data.startswith("http://") or image_data.startswith("https://"))

            if not image_data or is_placeholder or not is_data_url: # Only allow data URLs for now
                error_message = "Invalid or missing image data provided. Please provide a valid base64 encoded image data URL (e.g., 'data:image/jpeg;base64,...')."
                if is_placeholder:
                    error_message = "Placeholder image data received. Actual image data is required."
                logger.warning(f"BusinessCardTool validation failed: {error_message} (Input was: '{str(image_data)[:50]}...')")
                return json.dumps({"status": "error", "message": error_message})
            # --- End Input Validation ---

            # Construct the prompt for gpt-4o

            # Construct the prompt for gpt-4o
            prompt_messages = [
                SystemMessage(
                    content=f"""You are an expert business card scanner. Analyze the provided image of a business card and extract the contact information. Structure the output strictly as a JSON object following this Pydantic model format:

{self.output_parser.get_format_instructions()}

Extract only the information visible on the card. If a field is not present, omit it or set it to null/empty string as appropriate based on the format instructions. Do not infer information not present. Pay close attention to names, titles, company names, emails, phone numbers, websites, and any location details. Respond ONLY with the JSON object."""
                ),
                HumanMessage(
                    content=[
                        {"type": "text", "text": "Please extract the contact information from this business card image:"},
                        {
                            "type": "image_url",
                            "image_url": {
                                # Assuming JPEG, adjust if other formats are common
                                # Use the full data URL directly from input, assuming frontend sends it
                                "url": image_data
                            },
                        },
                    ]
                ),
            ]

            # Invoke the OpenAI client
            response = self.client.invoke(prompt_messages)
            response_content = str(response.content) # Ensure it's a string

            # Attempt to parse the JSON response using the Pydantic parser
            try:
                # The response should ideally be just the JSON string
                parsed_info: BusinessCardInfo = self.output_parser.parse(response_content)
                # Add status for consistency, although not part of BusinessCardInfo model
                result_dict = parsed_info.model_dump()
                result_dict["status"] = "success"
                return json.dumps(result_dict)
            except (json.JSONDecodeError, ValidationError) as e:
                logger.error(f"Failed to parse LLM response for business card: {e}", exc_info=True)
                # Return the raw response with an error note if parsing fails
                return json.dumps({
                    "status": "error",
                    "message": f"Failed to parse extracted information: {e}",
                    "raw_output": response_content
                })

        except Exception as e:
            # Catch any other unexpected errors during processing (e.g., API errors)
            logger.error(f"Error during business card scan: {e}", exc_info=True)
            return json.dumps({"status": "error", "message": f"Failed to process business card with OpenAI: {str(e)}"})

# Instantiate the tool for registration in the agent
# Initialization now happens within the __init__ method upon instantiation
business_card_tool = BusinessCardTool()

# Example usage (for testing purposes - requires OPENAI_API_KEY)
if __name__ == '__main__':
    # Configure basic logging for direct script execution
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info("--- Running business_card.py directly for testing ---")
    # Create a new instance for testing to ensure __init__ runs
    test_tool_instance = BusinessCardTool()

    if not test_tool_instance.enabled:
        logger.warning("Skipping BusinessCardTool test: Tool instance is disabled (check config/API key).")
        if test_tool_instance.init_error:
            logger.warning(f"Initialization error: {test_tool_instance.init_error}")
    else:
        logger.info("BusinessCardTool is enabled. Attempting test scan...")
        # You would need a real base64 encoded image string here
        # For example:
        try:
            # Provide a path to a real test image
            test_image_path = "path/to/your/test_business_card.jpg" # <--- CHANGE THIS PATH
            with open(test_image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            logger.info(f"Attempting scan with image: {test_image_path}")

            # Use the _run method of the test instance
            result_json = test_tool_instance._run(image_data=encoded_string)

            logger.info("\nScan Result (from test instance):")
            try:
                result_dict = json.loads(result_json)
                logger.info(json.dumps(result_dict, indent=2))
            except json.JSONDecodeError:
                logger.warning("Raw output (not valid JSON):")
                logger.warning(result_json)
        except FileNotFoundError:
             logger.error(f"Test image file not found at '{test_image_path}'. Skipping scan test.")
             logger.error("Please update the 'test_image_path' variable in the script.")
        except Exception as test_e:
             logger.error(f"Error during test execution: {test_e}", exc_info=True)

        # print("BusinessCardTool instance created. Run with actual image data for testing.") # Old message
