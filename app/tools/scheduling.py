import os
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Any
from googleapiclient.discovery import build, Resource
from google.oauth2 import service_account
from google.auth.exceptions import DefaultCredentialsError
from langchain.tools import tool

# Get a logger for this module
logger = logging.getLogger(__name__)
from langchain_core.callbacks import CallbackManagerForToolRun
import pytz

# Project components
from ..config import is_feature_enabled # Import config checker

# Check credentials only if feature is intended to be enabled
SCHEDULING_ENABLED = is_feature_enabled('scheduling.enabled')
if SCHEDULING_ENABLED and not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
    logger.warning("Scheduling feature is enabled but GOOGLE_APPLICATION_CREDENTIALS environment variable not set. Google Calendar API calls may fail.")

class SchedulingTool:
    """Tool for scheduling follow-up meetings via Google Calendar. Configurable via config.yaml."""

    def __init__(self):
        self.enabled = SCHEDULING_ENABLED
        self.calendar_service: Optional[Resource] = None
        self.calendar_id: str = os.getenv("GOOGLE_CALENDAR_ID", "primary")
        self.init_error: Optional[str] = None

        if self.enabled:
            credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            scopes = ['https://www.googleapis.com/auth/calendar']

            if credentials_path and os.path.exists(credentials_path):
                try:
                    credentials = service_account.Credentials.from_service_account_file(
                        credentials_path, scopes=scopes
                    )
                    self.calendar_service = build('calendar', 'v3', credentials=credentials)
                    logger.info("Google Calendar service initialized successfully for SchedulingTool.")
                except DefaultCredentialsError as e:
                    self.init_error = f"Google Credentials Error: {e}. Ensure the service account has permissions for the calendar."
                    logger.warning(f"{self.init_error}", exc_info=True)
                    self.enabled = False # Disable if client fails to init
                except Exception as e:
                    self.init_error = f"Error initializing Google Calendar service: {e}"
                    logger.warning(f"{self.init_error}", exc_info=True)
                    self.enabled = False # Disable if client fails to init
            else:
                self.init_error = "GOOGLE_APPLICATION_CREDENTIALS path not set or file not found."
                logger.warning(f"{self.init_error}")
                self.enabled = False # Disable if credentials missing
        else:
             logger.info("SchedulingTool (Google Calendar) is disabled by configuration.")


    @tool("schedule_follow_up_meeting")
    def schedule_follow_up_meeting(self, meeting_details_json: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """
        Schedules a follow-up meeting via Google Calendar.

        Args:
            meeting_details_json: JSON string with meeting details including:
                - summary (str): Title of the meeting.
                - description (str, optional): Meeting description.
                - start_time (str): ISO 8601 format start time (e.g., "2025-04-05T10:00:00+05:30").
                - duration_minutes (int, optional): Duration in minutes (default 30).
                - attendees (List[str], optional): List of attendee email addresses.
                - timezone (str, optional): Timezone for the event (e.g., "Asia/Calcutta", defaults to UTC if not specified in start_time).

        Returns:
            JSON string with confirmation message including meeting link, or indicating the feature is disabled or an error occurred.
        """
        if not self.enabled:
             return json.dumps({"status": "disabled", "message": "Scheduling feature is disabled in configuration."})
        if not self.calendar_service:
            # This case covers initialization errors after enabled check
            return json.dumps({"status": "error", "message": f"Google Calendar service not available. {self.init_error or ''}".strip()})

        try:
            # Parse meeting details
            try:
                meeting = json.loads(meeting_details_json)
            except json.JSONDecodeError:
                return json.dumps({"status": "error", "message": "Invalid JSON format for meeting details."})

            # Extract meeting information
            summary = meeting.get("summary")
            if not summary:
                return json.dumps({"status": "error", "message": "Meeting 'summary' (title) is required."})

            description = meeting.get("description", "")
            start_time_str = meeting.get("start_time")
            if not start_time_str:
                return json.dumps({"status": "error", "message": "Meeting 'start_time' (ISO 8601 format) is required."})

            duration = int(meeting.get("duration_minutes", 30))
            attendees_list = meeting.get("attendees", [])
            event_timezone = meeting.get("timezone") # Optional override

            # Parse start time and calculate end time
            try:
                start_dt = datetime.fromisoformat(start_time_str)
                # If timezone is naive, try to apply the provided one or default to UTC
                if start_dt.tzinfo is None or start_dt.tzinfo.utcoffset(start_dt) is None:
                    if event_timezone:

                        tz = pytz.timezone(event_timezone)
                        start_dt = tz.localize(start_dt)
                    else:
                         # Default to UTC if no timezone info provided
                         start_dt = start_dt.replace(tzinfo=timezone.utc)

            except (ValueError, TypeError) as e:
                return json.dumps({"status": "error", "message": f"Invalid start_time format: {e}. Use ISO 8601 format."})

            end_dt = start_dt + timedelta(minutes=duration)

            # Format attendees for Google Calendar API
            attendees_formatted = [{'email': email} for email in attendees_list if isinstance(email, str)]

            # Format event body for Google Calendar API
            event_body = {
                'summary': summary,
                'description': description,
                'start': {
                    'dateTime': start_dt.isoformat(),
                    'timeZone': str(start_dt.tzinfo), # Use the timezone from the datetime object
                },
                'end': {
                    'dateTime': end_dt.isoformat(),
                    'timeZone': str(end_dt.tzinfo),
                },
                'attendees': attendees_formatted,
                'reminders': {
                    'useDefault': False,
                    'overrides': [
                        {'method': 'email', 'minutes': 24 * 60}, # 1 day before
                        {'method': 'popup', 'minutes': 30},     # 30 mins before
                    ],
                },
                # Add conference data if needed (e.g., Google Meet)
                # 'conferenceData': {
                #     'createRequest': {
                #         'requestId': f"meeting-{datetime.now().timestamp()}", # Unique request ID
                #         'conferenceSolutionKey': {'type': 'hangoutsMeet'}
                #     }
                # }
            }

            # Create the event using the API
            created_event = self.calendar_service.events().insert(
                calendarId=self.calendar_id,
                body=event_body,
                sendUpdates='all', # Notify attendees
                # conferenceDataVersion=1 # Required if adding conference data
            ).execute()

            # Return confirmation with meeting link
            meeting_link = created_event.get('htmlLink', 'N/A')
            return json.dumps({
                "status": "success",
                "message": f"Meeting '{summary}' scheduled successfully.",
                "start_time": start_dt.isoformat(),
                "end_time": end_dt.isoformat(),
                "calendar_link": meeting_link
            })

        except Exception as e:
            # Log the full error: logging.exception("Google Calendar scheduling failed")
            # Check for specific API errors if possible
            error_details = str(e)
            if hasattr(e, 'content'): # Google API errors often have content
                try:
                    error_content = json.loads(e.content.decode('utf-8'))
                    error_details = error_content.get('error', {}).get('message', error_details)
                except Exception as parse_err:
                    logger.warning(f"Failed to parse Google API error content: {parse_err}")
                    pass # Keep original error string if content parsing fails
            logger.error(f"Error scheduling meeting: {error_details}", exc_info=True) # Log the detailed error
            return json.dumps({"status": "error", "message": f"Error scheduling meeting: {error_details}"})

# Example usage (for testing purposes)
if __name__ == '__main__':
    # Configure basic logging for direct script execution
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info("--- Running scheduling.py directly for testing ---")
    # This requires Google credentials to be set correctly
    # Ensure GOOGLE_APPLICATION_CREDENTIALS is set
    if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
         logger.warning("Skipping SchedulingTool test: Set GOOGLE_APPLICATION_CREDENTIALS environment variable.")
    else:
        tool_instance = SchedulingTool()

        # Check if service initialized before proceeding
        if tool_instance.calendar_service:
            # Example meeting details
            meeting_info = {
                "summary": "Test Follow-up Meeting with Lead",
                "description": "Discussing coffee blend options and pricing.",
                "start_time": (datetime.now(timezone.utc) + timedelta(days=3)).replace(hour=10, minute=0, second=0, microsecond=0).isoformat(),
                "duration_minutes": 45,
                "attendees": ["test_lead@example.com", os.getenv("SENDER_EMAIL", "default_user@example.com")], # Add attendee emails
                "timezone": "Asia/Calcutta" # Example timezone
            }
            meeting_json = json.dumps(meeting_info)

            logger.info("\nScheduling meeting...")
            result_json = tool_instance.schedule_follow_up_meeting(meeting_json)
            logger.info("\nScheduling Result:")
            try:
                # Pretty print the JSON result
                result_dict = json.loads(result_json)
                logger.info(json.dumps(result_dict, indent=2))
            except json.JSONDecodeError:
                logger.warning(f"Raw result (not valid JSON): {result_json}")
        else:
             logger.warning(f"Skipping test execution due to Calendar service initialization error: {tool_instance.init_error}")
