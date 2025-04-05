import React, { useState, useEffect } from 'react';
import './ConversationHistoryPanel.css'; // We'll create this CSS file next

const ConversationHistoryPanel = ({ onSelectSession, apiBaseUrl }) => {
  const [sessions, setSessions] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchSessions = async () => {
      setIsLoading(true);
      setError(null);
      try {
        const response = await fetch(`${apiBaseUrl}/sessions`); // Use prop for base URL
        if (!response.ok) {
          const errorData = await response.text();
          throw new Error(`HTTP error! status: ${response.status} - ${errorData}`);
        }
        const data = await response.json();
        // Assuming the endpoint returns an array of sessions directly
        if (Array.isArray(data)) {
          setSessions(data);
        } else {
          // Handle cases where the API might wrap the sessions array, e.g., { sessions: [...] }
          // Adjust this based on the actual API response structure if needed
          console.warn("Unexpected API response structure for sessions:", data);
          setSessions(data.sessions || []); // Attempt to access a 'sessions' key
        }
      } catch (err) {
        console.error("Failed to fetch conversation sessions:", err);
        setError(`Failed to load history: ${err.message}`);
      } finally {
        setIsLoading(false);
      }
    };

    fetchSessions();
  }, [apiBaseUrl]); // Re-fetch if apiBaseUrl changes (though unlikely)

  const formatTimestamp = (timestamp) => {
    if (!timestamp) return 'N/A';
    try {
      // Attempt to create a Date object. Handle potential ISO strings or other formats.
      const date = new Date(timestamp);
      // Check if the date is valid
      if (isNaN(date.getTime())) {
        return 'Invalid Date';
      }
      return date.toLocaleString(); // Adjust formatting as needed
    } catch (e) {
      console.error("Error formatting timestamp:", timestamp, e);
      return 'Invalid Date';
    }
  };


  return (
    <div className="conversation-history-panel">
      <h3>Conversation History</h3>
      {isLoading && <p>Loading history...</p>}
      {error && <p className="error-message">{error}</p>}
      {!isLoading && !error && sessions.length === 0 && <p>No conversation history found.</p>}
      {!isLoading && !error && sessions.length > 0 && (
        <ul className="session-list">
          {sessions.map((session) => (
            <li key={session.id} onClick={() => onSelectSession(session.id)} className="session-item">
              <div className="session-info">
                <span className="session-id">ID: {session.id.substring(0, 8)}...</span>
                {/* Display lead info if available, e.g., session.lead_name */}
                <span className="session-time">Started: {formatTimestamp(session.start_time)}</span>
              </div>
              {session.summary && <p className="session-summary">Summary: {session.summary}</p>}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
};

export default ConversationHistoryPanel;
