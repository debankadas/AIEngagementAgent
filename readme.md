# Implementation Guide: Trade Show Lead Management Agent

This guide provides detailed instructions for implementing, testing, and deploying the AI Interaction and Engagement Agent for trade show exhibitors.

## Table of Contents

1. [Setup and Installation](#setup-and-installation)
2. [Key Feature Implementation](#key-feature-implementation)
3. [Integration Points](#integration-points)
4. [Deployment Steps](#deployment-steps)
5. [Testing Strategy](#testing-strategy)
6. [User Guide](#user-guide)
7. [Limitations and Considerations](#limitations-and-considerations)
8. [Future Enhancements](#future-enhancements)

## Setup and Installation

### Environment Setup

1. **Create a new Python environment**:
   ```bash
   python -m venv leadagent-env
   source leadagent-env/bin/activate  # On Windows: leadagent-env\Scripts\activate
   ```

2. **Install required packages**:
   ```bash
   pip install -r requirements.txt
   ```

   Contents of `requirements.txt`:
   ```
   langchain==0.1.8
   langchain-anthropic==0.1.1
   langchain-openai==0.0.5
   langgraph==0.0.38
   fastapi==0.109.2
   uvicorn==0.27.1
   firebase-admin==6.3.0
   pytesseract==0.3.10
   Pillow==10.2.0
   python-multipart==0.0.6
   python-dotenv==1.0.0
   pydantic==2.5.3
   ```

3. **Set up environment variables**:
   Create a `.env` file with the following variables:
   ```
   ANTHROPIC_API_KEY=your_anthropic_api_key
   OPENAI_API_KEY=your_openai_api_key
   FIREBASE_CREDENTIALS_PATH=path/to/firebase-credentials.json
   GOOGLE_APPLICATION_CREDENTIALS=path/to/google-credentials.json
   ELEVENLABS_API_KEY=your_elevenlabs_api_key
   ```

### Project Structure

```
trade-show-lead-agent/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   ├── langraph_agent.py    # LangGraph agent implementation
│   ├── tools/               # Agent tools
│   │   ├── __init__.py
│   │   ├── business_card.py # Business card scanning tool
│   │   ├── database.py      # Database interaction tool
│   │   ├── email.py         # Email tool
│   │   ├── scheduling.py    # Meeting scheduling tool
│   │   └── transcription.py # Conversation analysis tool
│   ├── models/              # Pydantic models
│   │   ├── __init__.py
│   │   ├── lead.py          # Lead data models
│   │   └── conversation.py  # Conversation models
│   └── utils/               # Utility functions
│       ├── __init__.py
│       ├── prompts.py       # System prompts
│       └── helpers.py       # Helper functions
├── frontend/                # React frontend
│   ├── package.json
│   ├── public/
│   └── src/
│       ├── components/
│       ├── pages/
│       ├── App.js
│       └── index.js
├── tests/                   # Tests
│   ├── __init__.py
│   ├── test_langraph.py
│   ├── test_tools.py
│   └── test_api.py
├── .env                     # Environment variables
├── requirements.txt         # Python dependencies
├── Dockerfile               # Docker configuration
├── docker-compose.yml       # Docker Compose configuration
└── README.md                # Project documentation
```

## Key Feature Implementation

### 1. Business Card Scanning with OCR

The OCR implementation uses Google Vision API for accurate business card scanning:

```python
# app/tools/business_card.py
import io
import base64
from google.cloud import vision
from langchain.tools import tool
from langchain.callbacks.manager import CallbackManagerForToolRun
from typing import Optional

class BusinessCardTool:
    """Tool for scanning business cards and extracting contact information"""
    
    def __init__(self):
        self.client = vision.ImageAnnotatorClient()
    
    @tool("scan_business_card")
    def scan_business_card(self, image_data: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """
        Scan a business card image and extract contact information.
        
        Args:
            image_data: Base64 encoded string of the business card image
            
        Returns:
            JSON string with extracted contact information
        """
        try:
            # Decode the base64 image
            image_bytes = base64.b64decode(image_data)
            image = vision.Image(content=image_bytes)
            
            # Perform text detection
            response = self.client.text_detection(image=image)
            
            if response.error.message:
                return f"Error: {response.error.message}"
            
            # Extract text from the response
            texts = response.text_annotations
            if not texts:
                return '{"error": "No text found on the business card"}'
            
            # The first annotation contains all the text
            full_text = texts[0].description
            
            # Parse the text to extract structured information
            # This is a simplified implementation; production should use NER
            contact_info = self._parse_business_card_text(full_text)
            
            return json.dumps(contact_info)
        
        except Exception as e:
            return f'{{"error": "Failed to process business card: {str(e)}"}}'
    
    def _parse_business_card_text(self, text: str) -> dict:
        """
        Parse business card text into structured information.
        In production, this would use advanced NER models.
        """
        # Simple parsing based on common patterns
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        # Initialize empty contact info
        contact_info = {
            "full_name": "",
            "company": "",
            "role": "",
            "email": "",
            "phone": "",
            "website": "",
            "location": ""
        }
        
        # Basic parsing logic
        # First line is often the name
        if lines:
            contact_info["full_name"] = lines[0]
        
        # Look for email, phone, website
        for line in lines:
            if '@' in line and '.' in line and not contact_info["email"]:
                contact_info["email"] = line
            elif any(domain in line.lower() for domain in ['.com', '.org', '.net', '.io']) and not contact_info["website"]:
                contact_info["website"] = line
            elif any(p in line for p in ['+', '(', ')', '-']) and any(c.isdigit() for c in line) and not contact_info["phone"]:
                contact_info["phone"] = line
        
        # Second line is often the role
        if len(lines) > 1 and not any(key in lines[1].lower() for key in ['@', '.com', '+', '(']):
            contact_info["role"] = lines[1]
        
        # Company name may be in larger font or emphasized
        # This is a simplification; actual implementation would use font size/style analysis
        for i, line in enumerate(lines):
            if i > 1 and not any(key in line.lower() for key in ['@', '.com', '+', '(']):
                if not contact_info["company"]:
                    contact_info["company"] = line
                    break
        
        # For location, look for common address patterns
        # This is also simplified
        for line in lines:
            if any(word in line.lower() for word in ['road', 'street', 'ave', 'blvd', 'dr', 'lane']):
                contact_info["location"] = line
                break
        
        return contact_info
```

### 2. Multi-Language Support

Implement multi-language support using language detection and translation:

```python
# app/utils/language.py
from google.cloud import translate_v2 as translate
from langdetect import detect

class LanguageProcessor:
    def __init__(self):
        self.translate_client = translate.Client()
    
    def detect_language(self, text):
        """Detect the language of a given text"""
        try:
            return detect(text)
        except:
            return 'en'  # Default to English
    
    def translate_text(self, text, target_language='en'):
        """Translate text to target language"""
        if not text:
            return text
            
        source_language = self.detect_language(text)
        
        # If already in target language, return as is
        if source_language == target_language:
            return text
        
        result = self.translate_client.translate(
            text,
            target_language=target_language,
            source_language=source_language
        )
        
        return result['translatedText']
    
    def translate_conversation(self, messages, target_language='en'):
        """Translate all messages in a conversation"""
        translated_messages = []
        
        for message in messages:
            translated_content = self.translate_text(
                message['content'], 
                target_language
            )
            
            translated_messages.append({
                **message,
                'content': translated_content,
                'original_content': message['content'],
                'detected_language': self.detect_language(message['content'])
            })
        
        return translated_messages
```

### 3. Progressive AI Memory Implementation

This component allows the AI to learn and improve its knowledge of the exhibitor's products and business:

```python
# app/utils/memory.py
import json
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document

class ProgressiveMemory:
    def __init__(self, company_id, persistent_dir="./memory_storage"):
        self.company_id = company_id
        self.embeddings = OpenAIEmbeddings()
        self.persistent_dir = f"{persistent_dir}/{company_id}"
        
        # Initialize vector store
        self.vectorstore = Chroma(
            collection_name=f"company_{company_id}",
            embedding_function=self.embeddings,
            persist_directory=self.persistent_dir
        )
        
        # Text splitter for preprocessing
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
    
    def add_product_knowledge(self, product_info):
        """Add product information to the memory"""
        # Prepare documents
        product_text = json.dumps(product_info) if isinstance(product_info, dict) else product_info
        docs = self.text_splitter.create_documents([product_text])
        
        # Add metadata
        for doc in docs:
            doc.metadata["type"] = "product_info"
            doc.metadata["company_id"] = self.company_id
        
        # Add to vector store
        self.vectorstore.add_documents(docs)
        self.vectorstore.persist()
    
    def add_conversation_insights(self, conversation, lead_info):
        """Add conversation insights to memory"""
        # Create a summary document
        summary_text = f"Conversation with {lead_info.get('full_name', 'a lead')} from {lead_info.get('company', 'unknown company')}.\n"
        summary_text += f"They were interested in: {', '.join(lead_info.get('products_of_interest', ['unknown products']))}.\n"
        summary_text += f"Their concerns were: {', '.join(lead_info.get('concerns', ['no specific concerns']))}.\n"
        
        docs = self.text_splitter.create_documents([summary_text])
        
        # Add metadata
        for doc in docs:
            doc.metadata["type"] = "conversation_insight"
            doc.metadata["company_id"] = self.company_id
            doc.metadata["lead_id"] = lead_info.get("id", "unknown")
            doc.metadata["timestamp"] = lead_info.get("conversation_timestamp", "")
        
        # Add to vector store
        self.vectorstore.add_documents(docs)
        self.vectorstore.persist()
    
    def retrieve_relevant_knowledge(self, query, k=3):
        """Retrieve relevant knowledge based on a query"""
        results = self.vectorstore.similarity_search(query, k=k)
        return results
    
    def retrieve_product_knowledge(self, product_name, k=2):
        """Retrieve knowledge specific to a product"""
        results = self.vectorstore.similarity_search(
            product_name, 
            k=k,
            filter={"type": "product_info"}
        )
        return results
    
    def generate_business_context(self):
        """Generate a comprehensive business context from all stored knowledge"""
        # Retrieve all documents (would be paginated in production)
        all_docs = self.vectorstore.similarity_search("company information products offerings", k=10)
        
        context = {
            "company_info": "",
            "products_info": "",
            "customer_insights": ""
        }
        
        # Organize by document type
        for doc in all_docs:
            if doc.metadata.get("type") == "product_info":
                context["products_info"] += doc.page_content + "\n\n"
            elif doc.metadata.get("type") == "conversation_insight":
                context["customer_insights"] += doc.page_content + "\n\n"
            else:
                context["company_info"] += doc.page_content + "\n\n"
        
        return context
```

### 4. Lead Profile Generation

This tool extracts structured lead information from conversations:

```python
# app/tools/transcription.py
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_anthropic import ChatAnthropic
from ..models.lead import LeadProfile

class TranscriptionTool:
    """Tool for analyzing conversations and extracting structured lead information"""
    
    def __init__(self, model_name="claude-3-haiku-20240307"):
        self.llm = ChatAnthropic(model=model_name)
        self.output_parser = PydanticOutputParser(pydantic_object=LeadProfile)
    
    def analyze_conversation(self, conversation_history):
        """
        Analyze a conversation to extract structured lead information.
        
        Args:
            conversation_history: List of conversation messages
            
        Returns:
            LeadProfile object with structured information
        """
        # Format the conversation for analysis
        formatted_conversation = self._format_conversation(conversation_history)
        
        # Create the prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", """
You are an expert conversation analyzer for trade shows. Your task is to extract 
detailed information about a lead from a conversation transcript. 

Extract the following information, if present:
1. Basic contact information (name, company, role, email, phone, website, location)
2. Products or services they're interested in
3. Minimum order quantity (MOQ) details
4. Pain points or concerns they mentioned
5. Their level of interest in our products (Low, Medium, Medium-High, High)
6. Preferred contact method
7. Any specific follow-up actions needed

Format your response as a structured JSON object.
            """),
            ("human", f"""
Here is the conversation transcript:

{formatted_conversation}

Extract all relevant lead information from this conversation.
            """)
        ])
        
        # Get a structured response
        response = self.llm.invoke(prompt.format())
        
        # Parse the response into a LeadProfile object
        try:
            # Extract the JSON part from the response
            content = response.content
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = content[json_start:json_end]
                lead_profile = self.output_parser.parse(json_str)
                return lead_profile
            else:
                # Fallback if JSON not found
                lead_profile = LeadProfile(
                    full_name="Unknown",
                    company="Unknown",
                    role="Unknown"
                )
                return lead_profile
        except Exception as e:
            # Handle parsing errors
            print(f"Error parsing lead profile: {str(e)}")
            lead_profile = LeadProfile(
                full_name="Unknown",
                company="Unknown",
                role="Unknown"
            )
            return lead_profile
    
    def _format_conversation(self, conversation_history):
        """Format conversation history for analysis"""
        formatted_lines = []
        
        for msg in conversation_history:
            role = msg.get("role", "unknown")
            if role == "human":
                speaker = "Visitor"
            elif role == "assistant":
                speaker = "AI Assistant"
            else:
                speaker = "System"
                
            formatted_lines.append(f"{speaker}: {msg.get('content', '')}")
        
        return "\n".join(formatted_lines)
```

## Integration Points

### 1. Firebase/Database Integration

```python
# app/tools/database.py
import firebase_admin
from firebase_admin import credentials, firestore
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional
from langchain.tools import tool
from langchain.callbacks.manager import CallbackManagerForToolRun

class DatabaseTool:
    """Tool for interacting with Firebase/Firestore database"""
    
    def __init__(self):
        # Initialize Firebase if not already initialized
        if not firebase_admin._apps:
            cred_path = os.getenv("FIREBASE_CREDENTIALS_PATH")
            cred = credentials.Certificate(cred_path)
            firebase_admin.initialize_app(cred)
        
        self.db = firestore.client()
    
    @tool("save_lead_to_database")
    def save_lead_to_database(self, lead_data: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """
        Save lead information to the database.
        
        Args:
            lead_data: JSON string with lead information
            
        Returns:
            Confirmation message
        """
        try:
            # Parse the lead data
            lead_info = json.loads(lead_data)
            
            # Add timestamp
            lead_info["created_at"] = datetime.now()
            lead_info["updated_at"] = datetime.now()
            
            # Add to Firestore
            lead_ref = self.db.collection("leads").document()
            lead_info["id"] = lead_ref.id
            
            lead_ref.set(lead_info)
            
            return f"Successfully saved lead information for {lead_info.get('full_name', 'the visitor')} to the database with ID {lead_ref.id}."
        except Exception as e:
            return f"Error saving lead to database: {str(e)}"
    
    def get_lead_by_id(self, lead_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve lead information by ID.
        
        Args:
            lead_id: Lead document ID
            
        Returns:
            Lead information as a dictionary
        """
        try:
            lead_ref = self.db.collection("leads").document(lead_id)
            lead_doc = lead_ref.get()
            
            if lead_doc.exists:
                return lead_doc.to_dict()
            else:
                return None
        except Exception as e:
            print(f"Error retrieving lead: {str(e)}")
            return None
    
    def update_lead(self, lead_id: str, update_data: Dict[str, Any]) -> bool:
        """
        Update lead information.
        
        Args:
            lead_id: Lead document ID
            update_data: Data to update
            
        Returns:
            Success status
        """
        try:
            # Add updated timestamp
            update_data["updated_at"] = datetime.now()
            
            lead_ref = self.db.collection("leads").document(lead_id)
            lead_ref.update(update_data)
            
            return True
        except Exception as e:
            print(f"Error updating lead: {str(e)}")
            return False
    
    def save_conversation(self, lead_id: str, conversation_history: list) -> str:
        """
        Save conversation history.
        
        Args:
            lead_id: Associated lead ID
            conversation_history: List of conversation messages
            
        Returns:
            Conversation document ID
        """
        try:
            # Create conversation document
            conversation_data = {
                "lead_id": lead_id,
                "messages": conversation_history,
                "created_at": datetime.now(),
                "updated_at": datetime.now()
            }
            
            conv_ref = self.db.collection("conversations").document()
            conv_ref.set(conversation_data)
            
            # Update lead with conversation reference
            lead_ref = self.db.collection("leads").document(lead_id)
            lead_ref.update({
                "conversation_ids": firestore.ArrayUnion([conv_ref.id]),
                "updated_at": datetime.now()
            })
            
            return conv_ref.id
        except Exception as e:
            print(f"Error saving conversation: {str(e)}")
            return ""
```

### 2. Google Calendar Integration

```python
# app/tools/scheduling.py
from googleapiclient.discovery import build
from google.oauth2 import service_account
import json
import os
from datetime import datetime, timedelta
from typing import Optional
from langchain.tools import tool
from langchain.callbacks.manager import CallbackManagerForToolRun

class SchedulingTool:
    """Tool for scheduling follow-up meetings via Google Calendar"""
    
    def __init__(self):
        # Set up Google Calendar API client
        credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        scopes = ['https://www.googleapis.com/auth/calendar']
        
        credentials = service_account.Credentials.from_service_account_file(
            credentials_path, scopes=scopes
        )
        
        self.calendar_service = build('calendar', 'v3', credentials=credentials)
        self.calendar_id = os.getenv("GOOGLE_CALENDAR_ID", "primary")
    
    @tool("schedule_follow_up")
    def schedule_follow_up(self, details: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """
        Schedule a follow-up meeting via Google Calendar.
        
        Args:
            details: JSON string with meeting details
            
        Returns:
            Confirmation message with meeting details
        """
        try:
            # Parse meeting details
            meeting = json.loads(details)
            
            # Extract meeting information
            summary = meeting.get("summary", "Follow-up Meeting")
            description = meeting.get("description", "")
            start_time_str = meeting.get("start_time")
            duration = int(meeting.get("duration_minutes", 30))
            attendees = meeting.get("attendees", [])
            
            if not start_time_str:
                # Default to scheduling a meeting 3 days from now at 10 AM
                start_time = datetime.now() + timedelta(days=3)
                start_time = start_time.replace(hour=10, minute=0, second=0, microsecond=0)
            else:
                # Parse the provided datetime
                start_time = datetime.fromisoformat(start_time_str)
            
            # Calculate end time
            end_time = start_time + timedelta(minutes=duration)
            
            # Format for Google Calendar API
            event = {
                'summary': summary,
                'description': description,
                'start': {
                    'dateTime': start_time.isoformat(),
                    'timeZone': 'UTC',
                },
                'end': {
                    'dateTime': end_time.isoformat(),
                    'timeZone': 'UTC',
                },
                'attendees': [{'email': email} for email in attendees],
                'reminders': {
                    'useDefault': False,
                    'overrides': [
                        {'method': 'email', 'minutes': 24 * 60},
                        {'method': 'popup', 'minutes': 30},
                    ],
                },
            }
            
            # Create the event
            event = self.calendar_service.events().insert(
                calendarId=self.calendar_id, body=event, sendUpdates='all'
            ).execute()
            
            # Return confirmation with meeting link
            return f"Meeting scheduled for {start_time.strftime('%Y-%m-%d %H:%M')}. Calendar link: {event.get('htmlLink')}"
        
        except Exception as e:
            return f"Error scheduling meeting: {str(e)}"
```

### 3. Email Integration

```python
# app/tools/email.py
import os
import json
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional
from datetime import datetime, timedelta
from langchain.tools import tool
from langchain.callbacks.manager import CallbackManagerForToolRun

class EmailTool:
    """Tool for sending follow-up emails to leads"""
    
    def __init__(self):
        # Email configuration
        self.smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", 587))
        self.sender_email = os.getenv("SENDER_EMAIL")
        self.password = os.getenv("EMAIL_PASSWORD")
    
    @tool("send_email")
    def send_email(self, email_details: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """
        Send an email to a lead.
        
        Args:
            email_details: JSON string with email details
            
        Returns:
            Confirmation message
        """
        try:
            # Parse email details
            details = json.loads(email_details)
            
            recipient = details.get("recipient")
            subject = details.get("subject", "Follow-up from the trade show")
            body_text = details.get("body_text", "")
            body_html = details.get("body_html", "")
            schedule_time = details.get("schedule_time", None)
            
            if not recipient:
                return "Error: Recipient email address is required"
            
            # Check if email should be scheduled for later
            current_time = datetime.now()
            if schedule_time:
                schedule_datetime = datetime.fromisoformat(schedule_time)
                
                # If scheduled for future, return confirmation without sending
                if schedule_datetime > current_time:
                    # In production, would use a scheduling service
                    time_diff = schedule_datetime - current_time
                    hours = int(time_diff.total_seconds() / 3600)
                    return f"Email scheduled to be sent to {recipient} in {hours} hours"
            
            # Prepare email
            message = MIMEMultipart("alternative")
            message["Subject"] = subject
            message["From"] = self.sender_email
            message["To"] = recipient
            
            # Add text part
            message.attach(MIMEText(body_text, "plain"))
            
            # Add HTML part if provided
            if body_html:
                message.attach(MIMEText(body_html, "html"))
            
            # Send email
            context = ssl.create_default_context()
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls(context=context)
                server.login(self.sender_email, self.password)
                server.sendmail(
                    self.sender_email, recipient, message.as_string()
                )
            
            return f"Email sent successfully to {recipient}"
        
        except Exception as e:
            return f"Error sending email: {str(e)}"
```

## Deployment Steps

### Docker Deployment

1. **Build the Docker image**:

   ```dockerfile
   # Dockerfile
   FROM python:3.10-slim
   
   WORKDIR /app
   
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt
   
   COPY . .
   
   EXPOSE 8000
   
   CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
   ```

2. **Docker Compose setup**:

   ```yaml
   # docker-compose.yml
   version: '3'
   
   services:
     backend:
       build: .
       ports:
         - "8000:8000"
       env_file:
         - .env
       volumes:
         - ./checkpoints:/app/checkpoints
         - ./memory_storage:/app/memory_storage
       restart: unless-stopped
     
     frontend:
       build: ./frontend
       ports:
         - "3000:3000"
       depends_on:
         - backend
       restart: unless-stopped
   ```

3. **Deploy using Docker Compose**:

   ```bash
   docker-compose up -d
   ```

### Cloud Deployment (AWS)

1. **Create ECR repositories**:

   ```bash
   aws ecr create-repository --repository-name trade-show-backend
   aws ecr create-repository --repository-name trade-show-frontend
   ```

2. **Push Docker images to ECR**:

   ```bash
   # Login to ECR
   aws ecr get-login-password --region <region> | docker login --username AWS --password-stdin <account-id>.dkr.ecr.<region>.amazonaws.com
   
   # Tag and push backend
   docker build -t trade-show-backend .
   docker tag trade-show-backend:latest <account-id>.dkr.ecr.<region>.amazonaws.com/trade-show-backend:latest
   docker push <account-id>.dkr.ecr.<region>.amazonaws.com/trade-show-backend:latest
   
   # Tag and push frontend
   cd frontend
   docker build -t trade-show-frontend .
   docker tag trade-show-frontend:latest <account-id>.dkr.ecr.<region>.amazonaws.com/trade-show-frontend:latest
   docker push <account-id>.dkr.ecr.<region>.amazonaws.com/trade-show-frontend:latest
   ```

3. **Deploy using ECS Fargate**:

   Create a task definition, cluster, and service using AWS console or CLI.

### Vercel Deployment (Frontend)

For quick MVP deployment, the frontend can be deployed to Vercel:

```bash
cd frontend
vercel
```

## Testing Strategy

### Unit Tests

```python
# tests/test_tools.py
import unittest
from unittest.mock import patch, MagicMock
import json
import base64
from app.tools.business_card import BusinessCardTool
from app.tools.database import DatabaseTool
from app.tools.email import EmailTool
from app.tools.scheduling import SchedulingTool
from app.tools.transcription import TranscriptionTool

class TestBusinessCardTool(unittest.TestCase):
    @patch('app.tools.business_card.vision.ImageAnnotatorClient')
    def test_scan_business_card(self, mock_client):
        # Setup mock response
        mock_text = MagicMock()
        mock_text.description = "John Doe\nCEO\nAcme Inc.\njohn@acme.com\n+1 (123) 456-7890"
        
        mock_response = MagicMock()
        mock_response.text_annotations = [mock_text]
        mock_response.error.message = ""
        
        mock_client_instance = MagicMock()
        mock_client_instance.text_detection.return_value = mock_response
        mock_client.return_value = mock_client_instance
        
        # Create tool instance
        tool = BusinessCardTool()
        
        # Create test image
        dummy_image = base64.b64encode(b"dummy image data").decode("utf-8")
        
        # Test scan method
        result = tool.scan_business_card(dummy_image)
        result_dict = json.loads(result)
        
        # Assertions
        self.assertEqual(result_dict["full_name"], "John Doe")
        self.assertEqual(result_dict["role"], "CEO")
        self.assertEqual(result_dict["company"], "Acme Inc.")
        self.assertEqual(result_dict["email"], "john@acme.com")
        
        # Verify method calls
        mock_client_instance.text_detection.assert_called_once()

# Additional test classes for other tools
```

### Integration Tests

```python
# tests/test_api.py
from fastapi.testclient import TestClient
import json
from app.main import app

client = TestClient(app)

def test_init_session():
    response = client.post(
        "/api/init-session",
        json={
            "company_name": "Test Company",
            "event_name": "Test Event",
            "company_info": "Info about the company",
            "products_info": "Info about products"
        }
    )
    assert response.status_code == 200
    assert "session_id" in response.json()
    assert response.json()["status"] == "initialized"

def test_send_message():
    # First init a session
    init_response = client.post(
        "/api/init-session",
        json={
            "company_name": "Test Company",
            "event_name": "Test Event",
            "company_info": "Info about the company",
            "products_info": "Info about products"
        }
    )
    session_id = init_response.json()["session_id"]
    
    # Then send a message
    response = client.post(
        "/api/send-message",
        json={
            "session_id": session_id,
            "message": "Hello, I'm interested in your products"
        }
    )
    assert response.status_code == 200
    assert "response" in response.json()
    assert "current_stage" in response.json()
```

### End-to-End Tests

```python
# tests/test_e2e.py
import unittest
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

class TestLeadManagementE2E(unittest.TestCase):
    def setUp(self):
        self.driver = webdriver.Chrome()
        self.driver.get("http://localhost:3000")  # Frontend URL
    
    def tearDown(self):
        self.driver.quit()
    
    def test_conversation_flow(self):
        # Wait for page to load
        WebDriverWait(self.driver, 10).until(
            EC.presence_of_element_located((By.ID, "message-input"))
        )
        
        # Start conversation
        message_input = self.driver.find_element(By.ID, "message-input")
        send_button = self.driver.find_element(By.ID, "send-button")
        
        # Send initial message
        message_input.send_keys("Hello, I'm interested in your products")
        send_button.click()
        
        # Wait for response
        WebDriverWait(self.driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "ai-message"))
        )
        
        # Verify AI responded
        ai_messages = self.driver.find_elements(By.CLASS_NAME, "ai-message")
        self.assertGreaterEqual(len(ai_messages), 1)
        
        # Continue conversation
        message_input.send_keys("I'm looking for coffee beans for my cafe")
        send_button.click()
        
        # Wait for another response
        time.sleep(3)  # Allow time for response
        
        # Verify conversation progresses
        ai_messages = self.driver.find_elements(By.CLASS_NAME, "ai-message")
        self.assertGreaterEqual(len(ai_messages), 2)
```

## User Guide

### Exhibitor Setup

1. **Account Creation**:
   - Sign up at [app-url]/signup
   - Provide company name, contact info, and booth details
   - Subscribe to the appropriate plan

2. **Profile Setup**:
   - Add company description and product catalog
   - Upload product images and specifications
   - Define conversation goals and lead qualification criteria

3. **Device Setup**:
   - Set up a dedicated tablet at your booth
   - Install the application or open the web app
   - Log in with your exhibitor credentials
   - Position the tablet for easy visitor access

### Using the Application

1. **Starting a Conversation**:
   - Approach 1: Exhibitor initiates conversation, then hands off to AI when busy
   - Approach 2: Visitor scans QR code to start AI conversation
   - Approach 3: Exhibitor keeps AI listening during their conversation

2. **Business Card Scanning**:
   - Ask visitor to place card on tablet camera view
   - Tap "Scan Card" button
   - Review and approve captured information

3. **Conversation Management**:
   - Monitor AI-visitor conversation in real-time
   - Take over conversation at any point by clicking "Step In"
   - Add notes during conversation by clicking "Add Note"

4. **Post-Conversation**:
   - Review AI-generated lead profile
   - Approve or modify suggested follow-up tasks
   - Export leads to your CRM system

### Dashboard Navigation

1. **Leads Overview**:
   - View all captured leads
   - Filter by interest level, products, or date
   - Sort by conversation quality or follow-up status

2. **Conversation Insights**:
   - Review in-depth analysis of all conversations
   - See trending topics and common questions
   - Track conversion rates and engagement metrics

3. **Follow-up Management**:
   - Manage email templates and scheduling
   - Track follow-up statuses
   - Set reminders for pending follow-ups

4. **Analytics**:
   - Track booth performance metrics
   - Compare lead quality across shows
   - Analyze product interest and objection patterns

## Limitations and Considerations

1. **Privacy and Data Security**:
   - All conversations are stored and analyzed
   - Inform visitors about data collection
   - Implement proper security measures (encryption, access controls)
   - Comply with GDPR and other privacy regulations

2. **Technical Requirements**:
   - Stable internet connection required
   - Tablet with good quality camera for business card scanning
   - Sufficient battery power or access to power outlet

3. **Limitations of AI Assistance**:
   - AI cannot fully replace human interaction
   - Complex questions may require human intervention
   - Technical product details may need verification

4. **Audio Environment Considerations**:
   - Noisy trade show environments can affect speech recognition
   - Consider using a dedicated microphone for better results
   - Provide fallback options like text input

5. **Integration Constraints**:
   - CRM integration depends on available APIs
   - Some systems may require custom connectors
   - Data format compatibility may vary

## Future Enhancements

1. **Advanced Analytics**:
   - Sentiment analysis across all conversations
   - Predictive lead scoring based on conversation patterns
   - Competitor mention tracking and analysis

2. **Enhanced AI Capabilities**:
   - Real-time objection detection and suggested responses
   - Multilingual translation for international trade shows
   - Voice tone and emotion detection

3. **Expanded Integration**:
   - Direct integration with popular CRM platforms (Salesforce, HubSpot)
   - Marketing automation platform connections
   - E-commerce integration for direct ordering

4. **Interactive Product Showcasing**:
   - AR/VR product demonstrations triggered by conversation
   - Interactive product comparison tools
   - Dynamic pricing calculator based on requirements

5. **Team Collaboration**:
   - Multi-user access with role-based permissions
   - Real-time team notifications for hot leads
   - Collaborative follow-up assignment and tracking