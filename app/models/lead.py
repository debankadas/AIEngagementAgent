from typing import List, Optional
from pydantic import BaseModel, Field
from datetime import datetime

class LeadProfile(BaseModel):
    """Pydantic model for storing structured lead information."""
    id: Optional[str] = Field(None, description="Unique identifier for the lead (e.g., Firestore document ID)")
    full_name: Optional[str] = Field(None, description="Full name of the lead")
    company: Optional[str] = Field(None, description="Company name")
    role: Optional[str] = Field(None, description="Job title/role")
    email: Optional[str] = Field(None, description="Email address")
    phone: Optional[str] = Field(None, description="Phone number")
    location: Optional[str] = Field(None, description="Location/city")
    website: Optional[str] = Field(None, description="Website URL")
    products_of_interest: List[str] = Field(default_factory=list, description="Products the lead is interested in")
    moq: Optional[str] = Field(None, description="Minimum order quantity details")
    concerns: List[str] = Field(default_factory=list, description="Pain points or concerns mentioned")
    preferred_contact: Optional[str] = Field(None, description="Preferred contact method (e.g., WhatsApp, Email)")
    interest_level: Optional[str] = Field(None, description="Assessed interest level (e.g., Low, Medium, Medium-High, High)")
    follow_up_actions: List[str] = Field(default_factory=list, description="Specific follow-up actions needed")
    conversation_timestamp: Optional[datetime] = Field(None, description="Timestamp of the last significant interaction")
    event_name: Optional[str] = Field(None, description="Name of the event where the lead was captured")
    created_at: Optional[datetime] = Field(default_factory=datetime.now, description="Timestamp when the lead record was created")
    updated_at: Optional[datetime] = Field(default_factory=datetime.now, description="Timestamp when the lead record was last updated")
    conversation_ids: List[str] = Field(default_factory=list, description="List of associated conversation document IDs")
    custom_fields: Optional[dict] = Field(None, description="Dictionary for any additional custom fields")

class BusinessCardInfo(BaseModel):
    """Pydantic model for information extracted from a business card."""
    full_name: Optional[str] = None
    company: Optional[str] = None
    role: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    website: Optional[str] = None
    location: Optional[str] = None
    raw_text: Optional[str] = None # Store the full OCR text if needed
