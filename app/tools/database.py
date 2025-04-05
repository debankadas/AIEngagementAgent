import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
import firebase_admin

# Get a logger for this module
logger = logging.getLogger(__name__)
from firebase_admin import credentials, firestore
from langchain.tools import tool
from langchain_core.callbacks import CallbackManagerForToolRun

# Assuming models are in the parent directory structure
from ..models.lead import LeadProfile
from ..models.conversation import ConversationHistory, ConversationMessage
from ..config import is_feature_enabled # Import config checker

# Initialize Firebase Admin SDK only if the feature is enabled
DB_ENABLED = is_feature_enabled('database.enabled')
firebase_initialized = False
if DB_ENABLED and not firebase_admin._apps:
    cred_path = os.getenv("FIREBASE_CREDENTIALS_PATH")
    if cred_path and os.path.exists(cred_path):
        try:
            cred = credentials.Certificate(cred_path)
            firebase_admin.initialize_app(cred)
            firebase_initialized = True
            logger.info("Firebase Admin SDK initialized successfully for DatabaseTool.")
        except Exception as e:
            logger.warning(f"Error initializing Firebase Admin SDK: {e}. Ensure the credential file is valid. Database operations will fail.", exc_info=True)
    else:
        logger.warning("Database feature is enabled but FIREBASE_CREDENTIALS_PATH not set or file not found. Firebase operations will fail.")
elif not DB_ENABLED:
     logger.info("DatabaseTool is disabled by configuration.")


class DatabaseTool:
    """Tool for interacting with the Firebase Firestore database. Configurable via config.yaml."""

    def __init__(self):
        self.enabled = DB_ENABLED and firebase_initialized # Feature must be enabled AND SDK init successful
        self.db = None
        self.init_error = None

        if self.enabled:
            try:
                self.db = firestore.client()
                logger.info("Firestore client obtained for DatabaseTool.")
            except Exception as e:
                self.init_error = f"Error obtaining Firestore client: {e}"
                logger.warning(f"{self.init_error}", exc_info=True)
                self.enabled = False # Disable if client fails
        elif DB_ENABLED and not firebase_initialized:
             self.init_error = "Firebase Admin SDK failed to initialize."


    def _get_db(self):
        """Helper to get the Firestore client, checking enabled status and initialization."""
        if not self.enabled:
             raise ConnectionError(f"Database feature is disabled or not initialized. {self.init_error or ''}".strip())
        if not self.db:
             # This should ideally not happen if __init__ logic is correct, but as a safeguard:
             if firebase_initialized:
                 try:
                     self.db = firestore.client()
                 except Exception as e:
                      raise ConnectionError(f"Failed to get Firestore client post-init: {e}")
             else:
                 raise ConnectionError(f"Firebase SDK not initialized. {self.init_error or ''}".strip())
        return self.db

    @tool("save_lead_to_database")
    def save_lead_to_database(self, lead_data_json: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """
        Saves or updates lead information in the Firestore database.
        Expects a JSON string representing a LeadProfile object.
        If the lead_data contains an 'id', it attempts to update; otherwise, it creates a new lead.

        Args:
            lead_data_json: JSON string containing lead information (LeadProfile schema).

        Returns:
            JSON string with confirmation message including the lead ID, or indicating the feature is disabled or an error occurred.
        """
        if not self.enabled:
            return json.dumps({"status": "disabled", "message": "Database feature is disabled in configuration."})

        try:
            db = self._get_db() # Will raise ConnectionError if not available
            lead_data = json.loads(lead_data_json)

            # Convert relevant string dates back to datetime objects if needed (Pydantic might handle this)
            # Example: lead_data['created_at'] = datetime.fromisoformat(lead_data['created_at'])

            lead_id = lead_data.get("id")
            lead_data["updated_at"] = datetime.now() # Always update timestamp

            if lead_id:
                # Update existing lead
                lead_ref = db.collection("leads").document(lead_id)
                # Remove ID from data before updating to avoid writing it to the document fields
                update_data = {k: v for k, v in lead_data.items() if k != 'id'}
                lead_ref.update(update_data)
                action = "updated"
            else:
                # Create new lead
                lead_data["created_at"] = lead_data.get("created_at", datetime.now()) # Set created_at if new
                lead_ref = db.collection("leads").document()
                lead_id = lead_ref.id
                lead_data["id"] = lead_id # Add the generated ID to the data
                lead_ref.set(lead_data)
                action = "saved"

            return json.dumps({
                "status": "success",
                "message": f"Successfully {action} lead information for {lead_data.get('full_name', 'N/A')} to the database.",
                "lead_id": lead_id
            })
        except ConnectionError as ce:
             logger.error(f"Database connection error in save_lead: {ce}", exc_info=True)
             return json.dumps({"status": "error", "message": str(ce)})
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in save_lead: {e}. Input: {lead_data_json}", exc_info=True)
            return json.dumps({"status": "error", "message": "Invalid JSON format for lead data."})
        except Exception as e:
            logger.error(f"Unexpected error saving lead to database: {e}", exc_info=True)
            return json.dumps({"status": "error", "message": f"Error saving lead to database: {str(e)}"})

    @tool("get_lead_from_database")
    def get_lead_from_database(self, lead_id: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """
        Retrieves lead information from Firestore by ID.

        Args:
            lead_id: The document ID of the lead to retrieve.

        Returns:
            JSON string representing the LeadProfile object, or indicating disabled/error status.
        """
        if not self.enabled:
            return json.dumps({"status": "disabled", "message": "Database feature is disabled in configuration."})

        try:
            db = self._get_db() # Will raise ConnectionError if not available
            lead_ref = db.collection("leads").document(lead_id)
            lead_doc = lead_ref.get()

            if lead_doc.exists:
                lead_data = lead_doc.to_dict()
                # Ensure datetime objects are handled correctly if needed for Pydantic validation
                # lead_data['created_at'] = lead_data['created_at'].isoformat() if isinstance(lead_data.get('created_at'), datetime) else None
                # lead_data['updated_at'] = lead_data['updated_at'].isoformat() if isinstance(lead_data.get('updated_at'), datetime) else None
                # lead_data['conversation_timestamp'] = lead_data['conversation_timestamp'].isoformat() if isinstance(lead_data.get('conversation_timestamp'), datetime) else None

                # Validate with Pydantic model before returning (optional but good practice)
                # validated_lead = LeadProfile(**lead_data)
                # return validated_lead.model_dump_json()
                return json.dumps(lead_data, default=str) # Use default=str to handle datetimes
            else:
                return json.dumps({"status": "not_found", "message": f"Lead with ID {lead_id} not found."})
        except ConnectionError as ce:
             logger.error(f"Database connection error in get_lead: {ce}", exc_info=True)
             return json.dumps({"status": "error", "message": str(ce)})
        except Exception as e:
            logger.error(f"Unexpected error retrieving lead {lead_id}: {e}", exc_info=True)
            return json.dumps({"status": "error", "message": f"Error retrieving lead: {str(e)}"})

    @tool("save_conversation_to_database")
    def save_conversation_to_database(self, conversation_data_json: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """
        Saves conversation history to Firestore.
        Expects a JSON string representing a ConversationHistory object.

        Args:
            conversation_data_json: JSON string containing conversation history (ConversationHistory schema).

        Returns:
            JSON string with confirmation message including the conversation ID, or indicating disabled/error status.
        """
        if not self.enabled:
            return json.dumps({"status": "disabled", "message": "Database feature is disabled in configuration."})

        try:
            db = self._get_db() # Will raise ConnectionError if not available
            conv_data = json.loads(conversation_data_json)

            # Validate lead_id exists
            lead_id = conv_data.get("lead_id")
            if not lead_id:
                return json.dumps({"status": "error", "message": "Missing 'lead_id' in conversation data."})

            # Convert message timestamps if needed
            # for msg in conv_data.get('messages', []):
            #     if isinstance(msg.get('timestamp'), str):
            #         msg['timestamp'] = datetime.fromisoformat(msg['timestamp'])

            conv_data["updated_at"] = datetime.now()

            # Create conversation document
            conv_ref = db.collection("conversations").document()
            conv_id = conv_ref.id
            conv_data["id"] = conv_id # Add generated ID
            conv_data["created_at"] = conv_data.get("created_at", datetime.now())

            conv_ref.set(conv_data)

            # Update the corresponding lead document to include this conversation ID
            lead_ref = db.collection("leads").document(lead_id)
            lead_ref.update({
                "conversation_ids": firestore.ArrayUnion([conv_id]),
                "updated_at": datetime.now()
            })

            return json.dumps({
                "status": "success",
                "message": f"Successfully saved conversation history for lead {lead_id}.",
                "conversation_id": conv_id
            })
        except ConnectionError as ce:
             logger.error(f"Database connection error in save_conversation: {ce}", exc_info=True)
             return json.dumps({"status": "error", "message": str(ce)})
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in save_conversation: {e}. Input: {conversation_data_json}", exc_info=True)
            return json.dumps({"status": "error", "message": "Invalid JSON format for conversation data."})
        except Exception as e:
            logger.error(f"Unexpected error saving conversation: {e}", exc_info=True)
            return json.dumps({"status": "error", "message": f"Error saving conversation: {str(e)}"})

    @tool("get_conversation_sessions")
    def get_conversation_sessions(self, limit: int = 50, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """
        Retrieves a list of recent conversation session summaries from Firestore.

        Args:
            limit: The maximum number of sessions to retrieve (default 50).

        Returns:
            JSON string containing a list of conversation summaries (id, start_time, lead_id, summary),
            or indicating disabled/error status.
        """
        if not self.enabled:
            return json.dumps({"status": "disabled", "message": "Database feature is disabled in configuration."})

        try:
            db = self._get_db()
            sessions_query = db.collection("conversations").order_by("start_time", direction=firestore.Query.DESCENDING).limit(limit)
            docs = sessions_query.stream()

            sessions = []
            for doc in docs:
                data = doc.to_dict()
                sessions.append({
                    "id": doc.id,
                    "start_time": data.get("start_time"),
                    "lead_id": data.get("lead_id"),
                    "summary": data.get("summary") # Include summary if available
                    # Add lead name here if we denormalize it onto the conversation record later
                })

            return json.dumps({"status": "success", "sessions": sessions}, default=str) # Use default=str for datetime
        except ConnectionError as ce:
             logger.error(f"Database connection error in get_conversation_sessions: {ce}", exc_info=True)
             return json.dumps({"status": "error", "message": str(ce)})
        except Exception as e:
            logger.error(f"Unexpected error retrieving conversation sessions: {e}", exc_info=True)
            return json.dumps({"status": "error", "message": f"Error retrieving conversation sessions: {str(e)}"})

    @tool("get_conversation_messages")
    def get_conversation_messages(self, conversation_id: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """
        Retrieves the full message history for a specific conversation ID from Firestore.

        Args:
            conversation_id: The document ID of the conversation to retrieve.

        Returns:
            JSON string representing the full ConversationHistory object (including messages),
            or indicating disabled/not_found/error status.
        """
        if not self.enabled:
            return json.dumps({"status": "disabled", "message": "Database feature is disabled in configuration."})

        try:
            db = self._get_db()
            conv_ref = db.collection("conversations").document(conversation_id)
            conv_doc = conv_ref.get()

            if conv_doc.exists:
                conv_data = conv_doc.to_dict()
                # Ensure messages are present, default to empty list if not
                if 'messages' not in conv_data:
                    conv_data['messages'] = []
                # Validate with Pydantic model before returning (optional)
                # validated_conv = ConversationHistory(**conv_data)
                # return validated_conv.model_dump_json()
                return json.dumps({"status": "success", "conversation": conv_data}, default=str) # Use default=str for datetime
            else:
                return json.dumps({"status": "not_found", "message": f"Conversation with ID {conversation_id} not found."})
        except ConnectionError as ce:
             logger.error(f"Database connection error in get_conversation_messages: {ce}", exc_info=True)
             return json.dumps({"status": "error", "message": str(ce)})
        except Exception as e:
            logger.error(f"Unexpected error retrieving conversation messages for ID {conversation_id}: {e}", exc_info=True)
            return json.dumps({"status": "error", "message": f"Error retrieving conversation messages: {str(e)}"})


# Example usage (for testing purposes)
if __name__ == '__main__':
    # Configure basic logging for direct script execution
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info("--- Running database.py directly for testing ---")
    # This requires Firebase credentials to be set correctly
    # Ensure FIREBASE_CREDENTIALS_PATH is set in your environment

    db_tool = DatabaseTool()

    # Example: Save a new lead
    new_lead = LeadProfile(
        full_name="Test Lead",
        company="Test Inc.",
        role="Tester",
        email="test@example.com",
        products_of_interest=["Test Product"],
        event_name="Test Event"
    )
    lead_json = new_lead.model_dump_json()
    # print(f"Saving lead: {lead_json}")
    # save_result = db_tool.save_lead_to_database(lead_json)
    # print(f"Save result: {save_result}")

    # Example: Get the lead (replace with actual ID from save_result if successful)
    # lead_id_to_get = "replace_with_actual_id"
    # if lead_id_to_get != "replace_with_actual_id":
    #     get_result = db_tool.get_lead_from_database(lead_id_to_get)
    #     print(f"Get result: {get_result}")

    # Example: Save conversation
    # conv_history = ConversationHistory(
    #     lead_id=lead_id_to_get, # Use the ID from the saved lead
    #     session_id="test_session_123",
    #     messages=[
    #         ConversationMessage(role="human", content="Hello"),
    #         ConversationMessage(role="assistant", content="Hi there!")
    #     ]
    # )
    # conv_json = conv_history.model_dump_json()
    # print(f"Saving conversation: {conv_json}")
    # save_conv_result = db_tool.save_conversation_to_database(conv_json)
    # print(f"Save conversation result: {save_conv_result}")
    pass # Avoid running actual DB operations without explicit action
