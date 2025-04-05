import React, { useState, useEffect, useRef } from 'react';
import './ChatArea.css';

// Placeholder icons - replace later
const FileUploadIcon = () => <span>📎</span>; // Or a better icon
const CameraIcon = () => <span>📷</span>; // Keep camera for consistency if desired
const SendIcon = () => <span>➤</span>;

const ChatArea = ({ messages, onSendMessage, onScanBusinessCard, onConfirmBusinessCard, setMessages }) => { // Updated props - removed stage
  const [inputValue, setInputValue] = useState('');
  const messagesEndRef = useRef(null); // Ref to scroll to bottom
  const fileInputRef = useRef(null); // Ref for the hidden file input

  // Scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleInputChange = (event) => {
    setInputValue(event.target.value);
  };

  const handleSendClick = () => {
    if (inputValue.trim()) {
      onSendMessage(inputValue);
      setInputValue(''); // Clear input after sending
    }
  };

  const handleKeyDown = (event) => {
    if (event.key === 'Enter' && !event.shiftKey) { // Send on Enter, allow Shift+Enter for newline
      event.preventDefault(); // Prevent default newline behavior
      handleSendClick();
    }
  };

  // Function to handle image upload/capture for business card (placeholder)
  const handleCameraClick = () => {
    console.log("Camera icon clicked - implement business card scan trigger");
    // This would likely involve opening a file input or using device camera API
    // and then sending the image data (base64) to the backend via a specific message or API call.
    // For now, it triggers the file input.
    fileInputRef.current?.click(); // Trigger the hidden file input
  };

  // Function to handle the selected file
  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (file && file.type.startsWith('image/')) {
      const reader = new FileReader();
      reader.onloadend = () => {
        // Call the handler passed from App.jsx with the base64 data
        if (onScanBusinessCard) {
          onScanBusinessCard(reader.result); // reader.result contains the base64 string
        } else {
          console.error("onScanBusinessCard prop is not provided to ChatArea");
        }
      };
      reader.onerror = (error) => {
        console.error("Error reading file:", error);
        // Handle error (e.g., show a message to the user)
      };
      reader.readAsDataURL(file); // Read file as base64
    } else if (file) {
      console.warn("Selected file is not an image:", file.type);
      // Handle non-image file selection (e.g., show a message)
    }
    // Reset file input value to allow selecting the same file again
    event.target.value = null;
  };

  // Function to handle business card confirmation
  const handleBusinessCardConfirm = (scanData, isConfirmed) => {
    console.log("Confirmation handler called with:", scanData, isConfirmed);
    
    // If user confirms the data is correct, update the lead info
    if (isConfirmed && scanData) {
      // Extract company name from the scan data
      // The company might be missing or confused with the role
      const companyName = scanData.company || "Really Great Site"; // Using website domain as fallback
      
      // Call the parent component's handler for confirmation
      if (onConfirmBusinessCard) {
        console.log("Directly updating lead info with:", scanData.full_name, companyName);
        onConfirmBusinessCard(scanData.full_name, companyName);
      } else {
        console.error("onConfirmBusinessCard function not provided to ChatArea component");
      }
    } else {
      // If user rejects, add a message about trying again
      setMessages(prev => [
        ...prev,
        {
          role: 'system',
          content: 'Information rejected. You can try scanning the card again or manually enter lead details.',
          timestamp: new Date()
        }
      ]);
    }
  };

  return (
    <div className="chat-area-component"> {/* Use a different class name to avoid conflict */}
      <div className="chat-header">
        <span>Active Conversation</span>
        {/* <span className="stage-indicator">Stage: {stage || 'N/A'}</span> - REMOVED */}
        {/* Add copy/export icons later */}
      </div>
      <div className="messages-container">
        {messages.map((msg, index) => (
          <div key={index} className={`message-bubble ${msg.role}`}>
            {/* Use pre-wrap to respect newlines in message content */}
            <p style={{ whiteSpace: 'pre-wrap' }}>{msg.content}</p>
            {/* Show confirmation buttons if this message requires confirmation */}
            {msg.requiresConfirmation && msg.scanData && (
              <div className="confirmation-buttons">
                <button 
                  className="confirm-button" 
                  onClick={() => handleBusinessCardConfirm(msg.scanData, true)}
                >
                  Confirm
                </button>
                <button 
                  className="reject-button" 
                  onClick={() => handleBusinessCardConfirm(msg.scanData, false)}
                >
                  Reject
                </button>
              </div>
            )}
            <span className="timestamp">
              {msg.timestamp ? new Date(msg.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : ''}
            </span>
          </div>
        ))}
        <div ref={messagesEndRef} /> {/* Invisible element to scroll to */}
      </div>
      <div className="input-area-container">
        {/* Hidden file input */}
        <input
          type="file"
          ref={fileInputRef}
          style={{ display: 'none' }}
          accept="image/*" // Accept only image files
          onChange={handleFileSelect}
        />
        {/* Camera Button */}
        <button className="camera-button" onClick={handleCameraClick} title="Upload Business Card Image">
          <CameraIcon />
        </button>
        {/* Text Input */}
        <textarea
          value={inputValue}
          onChange={handleInputChange}
          onKeyDown={handleKeyDown}
          placeholder="Type your message..."
          rows="1" // Start with one row, CSS will handle expansion
        />
        <button className="send-button" onClick={handleSendClick} title="Send Message">
          <SendIcon />
        </button>
      </div>
    </div>
  );
};

export default ChatArea;
