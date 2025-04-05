from typing import List, Optional, Literal
from pydantic import BaseModel, Field
from datetime import datetime

class ConversationMessage(BaseModel):
    """Pydantic model for a single message in a conversation."""
    role: Literal["human", "assistant", "system", "tool"] = Field(description="The role of the message sender")
    content: str = Field(description="The text content of the message")
    timestamp: datetime = Field(default_factory=datetime.now, description="Timestamp of the message")
    tool_call_id: Optional[str] = Field(None, description="ID if this message is a tool call result")
    # Add other relevant fields like language, sentiment, etc. if needed later

class ConversationHistory(BaseModel):
    """Pydantic model for storing the history of a conversation."""
    id: Optional[str] = Field(None, description="Unique identifier for the conversation (e.g., Firestore document ID)")
    lead_id: str = Field(description="ID of the lead associated with this conversation")
    session_id: str = Field(description="ID of the session this conversation belongs to")
    messages: List[ConversationMessage] = Field(default_factory=list, description="List of messages in the conversation")
    start_time: datetime = Field(default_factory=datetime.now, description="Timestamp when the conversation started")
    end_time: Optional[datetime] = Field(None, description="Timestamp when the conversation ended")
    summary: Optional[str] = Field(None, description="AI-generated summary of the conversation")
    tags: List[str] = Field(default_factory=list, description="Tags for categorizing the conversation")
    created_at: Optional[datetime] = Field(default_factory=datetime.now, description="Timestamp when the conversation record was created")
    updated_at: Optional[datetime] = Field(default_factory=datetime.now, description="Timestamp when the conversation record was last updated")
