import os
import json
import smtplib
import ssl
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional, List, Dict, Any

# Get a logger for this module
logger = logging.getLogger(__name__)
from langchain.tools import tool
from langchain_core.callbacks import CallbackManagerForToolRun
from email_validator import validate_email, EmailNotValidError # For basic validation

# Project components
from ..config import is_feature_enabled # Import config checker

# Check credentials only if feature is intended to be enabled
EMAIL_ENABLED = is_feature_enabled('email.enabled')
required_env_vars = ["SMTP_SERVER", "SMTP_PORT", "SENDER_EMAIL", "EMAIL_PASSWORD"]
missing_vars = []
if EMAIL_ENABLED:
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        logger.warning(f"Email feature is enabled but missing environment variables: {', '.join(missing_vars)}. Email tool may fail.")
else:
    logger.info("EmailTool is disabled by configuration.")


class EmailTool:
    """Tool for sending follow-up emails to leads via SMTP. Configurable via config.yaml."""

    def __init__(self):
        self.enabled = EMAIL_ENABLED and not missing_vars # Feature enabled and vars present
        self.smtp_server = os.getenv("SMTP_SERVER")
        self.smtp_port_str = os.getenv("SMTP_PORT")
        self.sender_email = os.getenv("SENDER_EMAIL")
        self.password = os.getenv("EMAIL_PASSWORD")
        self.init_error: Optional[str] = None
        self.smtp_port: Optional[int] = None

        if self.enabled:
            # Validate configuration further
            try:
                self.smtp_port = int(self.smtp_port_str)
            except (ValueError, TypeError):
                self.init_error = f"Invalid SMTP_PORT: '{self.smtp_port_str}'. Must be an integer."
                logger.error(f"{self.init_error}")
                self.enabled = False # Disable if port invalid

            if self.enabled: # Check again after port validation
                try:
                    # Validate sender email format
                    validate_email(self.sender_email, check_deliverability=False)
                    logger.info("EmailTool initialized successfully.")
                except EmailNotValidError as e:
                    self.init_error = f"Invalid SENDER_EMAIL format: {e}"
                    logger.error(f"{self.init_error}", exc_info=True)
                    self.enabled = False # Disable if sender email invalid
        elif EMAIL_ENABLED and missing_vars:
             self.init_error = f"Missing one or more SMTP environment variables: {', '.join(missing_vars)}"


    @tool("send_follow_up_email")
    def send_follow_up_email(self, email_details_json: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """
        Sends a follow-up email to a lead using configured SMTP settings.

        Args:
            email_details_json: JSON string with email details including:
                - recipient_email (str): Email address of the recipient.
                - subject (str): Subject line of the email.
                - body_text (str, optional): Plain text body of the email.
                - body_html (str, optional): HTML body of the email (provide at least one body type).
                - cc_emails (List[str], optional): List of CC email addresses.
                - bcc_emails (List[str], optional): List of BCC email addresses.

        Returns:
            JSON string confirming success, indicating disabled status, or detailing the error.
        """
        if not self.enabled:
             return json.dumps({"status": "disabled", "message": f"Email feature is disabled or not configured correctly. {self.init_error or ''}".strip()})
        # We already validated port is int in __init__ if enabled
        if self.smtp_port is None:
             # This case should theoretically not be reached if self.enabled is True, but as a safeguard:
             return json.dumps({"status": "error", "message": "SMTP port configuration error."})

        try:
            # Parse email details
            try:
                details = json.loads(email_details_json)
            except json.JSONDecodeError:
                return json.dumps({"status": "error", "message": "Invalid JSON format for email details."})

            recipient = details.get("recipient_email")
            subject = details.get("subject")
            body_text = details.get("body_text")
            body_html = details.get("body_html")
            cc_emails = details.get("cc_emails", [])
            bcc_emails = details.get("bcc_emails", [])

            # Validate inputs
            if not recipient:
                return json.dumps({"status": "error", "message": "Recipient email address ('recipient_email') is required."})
            if not subject:
                return json.dumps({"status": "error", "message": "Email 'subject' is required."})
            if not body_text and not body_html:
                return json.dumps({"status": "error", "message": "Email body ('body_text' or 'body_html') is required."})

            try:
                validate_email(recipient, check_deliverability=False) # Basic format check
                valid_cc = [validate_email(cc, check_deliverability=False).normalized for cc in cc_emails if cc]
                valid_bcc = [validate_email(bcc, check_deliverability=False).normalized for bcc in bcc_emails if bcc]
            except EmailNotValidError as e:
                return json.dumps({"status": "error", "message": f"Invalid email address format: {e}"})

            # Prepare email message
            message = MIMEMultipart("alternative")
            message["Subject"] = subject
            message["From"] = self.sender_email
            message["To"] = recipient
            if valid_cc:
                message["Cc"] = ", ".join(valid_cc)

            # Attach parts
            if body_text:
                message.attach(MIMEText(body_text, "plain", "utf-8"))
            if body_html:
                message.attach(MIMEText(body_html, "html", "utf-8"))

            # Combine all recipients for the sendmail command
            all_recipients = [recipient] + valid_cc + valid_bcc

            # Send email using SMTP
            context = ssl.create_default_context()
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                # The 'with' statement handles server.quit() automatically
                server.ehlo() # Identify ourselves to the server
                server.starttls(context=context) # Secure the connection
                server.ehlo() # Re-identify ourselves over TLS connection
                server.login(self.sender_email, self.password)
                server.sendmail(
                    self.sender_email, all_recipients, message.as_string()
                )
                logger.info(f"Email sent successfully to {recipient}") # Log success

            return json.dumps({"status": "success", "message": f"Email sent successfully to {recipient}."})

        except smtplib.SMTPAuthenticationError as auth_err:
             logger.error(f"SMTP Authentication failed. Check SENDER_EMAIL and EMAIL_PASSWORD. Error: {auth_err}", exc_info=True)
             return json.dumps({"status": "error", "message": "SMTP Authentication failed. Check credentials."})
        except smtplib.SMTPException as smtp_err:
             # Catch other SMTP errors (connection, sending, etc.)
             logger.error(f"SMTP Error: {smtp_err}", exc_info=True)
             return json.dumps({"status": "error", "message": f"SMTP Error: {smtp_err}"})
        except Exception as e:
            logger.error(f"Unexpected error sending email: {e}", exc_info=True)
            return json.dumps({"status": "error", "message": f"Error sending email: {str(e)}"})

# Example usage (for testing purposes)
if __name__ == '__main__':
    # Configure basic logging for direct script execution
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info("--- Running email.py directly for testing ---")
    # This requires SMTP environment variables to be set
    if missing_vars:
        logger.warning("Skipping EmailTool test: Set all required SMTP environment variables.")
    else:
        tool_instance = EmailTool()

        # Check if tool initialized correctly
        if tool_instance.enabled:
            # Example email details
            email_info = {
                "recipient_email": "test_recipient@example.com", # Replace with a real test address if desired
                "subject": "Test Follow-up from Trade Show Agent",
                "body_text": "Hello,\n\nThis is a test follow-up email.\n\nBest regards,\nAI Agent",
                "body_html": "<p>Hello,</p><p>This is a test follow-up email.</p><p>Best regards,<br>AI Agent</p>",
                "cc_emails": ["cc_test@example.com"],
                "bcc_emails": ["bcc_test@example.com"]
            }
            email_json = json.dumps(email_info)

            logger.info("\nSending email...")
            result_json = tool_instance.send_follow_up_email(email_json)
            logger.info("\nEmail Sending Result:")
            try:
                # Pretty print the JSON result
                result_dict = json.loads(result_json)
                logger.info(json.dumps(result_dict, indent=2))
            except json.JSONDecodeError:
                logger.warning(f"Raw result (not valid JSON): {result_json}")
        else:
             logger.warning(f"Skipping test execution due to Email tool initialization error or config: {tool_instance.init_error or 'Disabled in config'}")
