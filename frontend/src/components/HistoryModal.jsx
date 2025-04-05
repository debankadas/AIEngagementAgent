import React from 'react';
import './HistoryModal.css'; // We'll create this CSS file next

const HistoryModal = ({ isOpen, onClose, conversation, isLoading, error }) => {
  if (!isOpen) {
    return null;
  }

  const formatTimestamp = (timestamp) => {
    if (!timestamp) return '';
    try {
      const date = new Date(timestamp);
      if (isNaN(date.getTime())) return '';
      return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    } catch (e) {
      return '';
    }
  };

  return (
    <div className="history-modal-overlay" onClick={onClose}>
      <div className="history-modal-content" onClick={(e) => e.stopPropagation()}>
        <button className="history-modal-close-button" onClick={onClose}>
          &times; {/* Close icon */}
        </button>
        <h3>Conversation Details</h3>
        {isLoading && <p>Loading messages...</p>}
        {error && <p className="error-message">{error}</p>}
        {!isLoading && !error && conversation && (
          <>
            <div className="history-modal-header">
              <p><strong>Session ID:</strong> {conversation.id?.substring(0, 8)}...</p>
              <p><strong>Lead ID:</strong> {conversation.lead_id}</p>
              <p><strong>Started:</strong> {new Date(conversation.start_time).toLocaleString()}</p>
              {conversation.end_time && <p><strong>Ended:</strong> {new Date(conversation.end_time).toLocaleString()}</p>}
              {conversation.summary && <p><strong>Summary:</strong> {conversation.summary}</p>}
            </div>
            <div className="history-modal-messages-container">
              {conversation.messages && conversation.messages.length > 0 ? (
                conversation.messages.map((msg, index) => (
                  <div key={index} className={`history-message-bubble ${msg.role}`}>
                    {/* Use pre-wrap to respect newlines */}
                    <p style={{ whiteSpace: 'pre-wrap' }}>{msg.content}</p>
                    <span className="history-timestamp">
                      {formatTimestamp(msg.timestamp)}
                    </span>
                  </div>
                ))
              ) : (
                <p>No messages found for this conversation.</p>
              )}
            </div>
          </>
        )}
         {!isLoading && !error && !conversation && (
            <p>No conversation data available.</p>
         )}
      </div>
    </div>
  );
};

export default HistoryModal;
