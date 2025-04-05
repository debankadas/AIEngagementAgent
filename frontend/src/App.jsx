import React, { useState, useEffect, useRef, useCallback } from 'react';
import { FaMoon, FaSun } from 'react-icons/fa';
import './App.css';
import Sidebar from './components/Sidebar';
import ChatArea from './components/ChatArea';
import OverviewPanel from './components/OverviewPanel';
import ConversationHistoryPanel from './components/ConversationHistoryPanel';
import HistoryModal from './components/HistoryModal';
import CompanySelector from './components/CompanySelector'; // Import the new component
import './components/ConversationHistoryPanel.css';
import './components/HistoryModal.css';

function App() {
  const [sessionId, setSessionId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [isConnected, setIsConnected] = useState(false);
  const [isConnecting, setIsConnecting] = useState(false);
  const [error, setError] = useState(null);
  const [showCompanySelector, setShowCompanySelector] = useState(true); // Start with company selector visible
  const [selectedCompany, setSelectedCompany] = useState(null); // Store the selected company
  const [isDarkMode, setIsDarkMode] = useState(true); // State for theme - dark mode as default
  const [leadInfo, setLeadInfo] = useState({
      startTime: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      full_name: null, // Initialize fields from LeadProfile
      company: null,
      // Add other relevant fields you want to display in OverviewPanel
  });
  
  // Toggle dark mode
  const toggleDarkMode = () => {
    setIsDarkMode(!isDarkMode);
  };

  // Apply dark mode class to document root
  useEffect(() => {
    if (isDarkMode) {
      document.documentElement.classList.add('dark-theme');
    } else {
      document.documentElement.classList.remove('dark-theme');
    }
  }, [isDarkMode]);

  // Log lead info changes
  useEffect(() => {
    console.log("Lead info updated:", leadInfo);
  }, [leadInfo]);
  const [activeView, setActiveView] = useState('chat'); // 'chat' or 'history'
  const [isHistoryModalOpen, setIsHistoryModalOpen] = useState(false);
  const [selectedHistoryConversation, setSelectedHistoryConversation] = useState(null);
  const [isHistoryLoading, setIsHistoryLoading] = useState(false);
  const [historyError, setHistoryError] = useState(null);

  const ws = useRef(null);

  const API_BASE_URL = 'http://localhost:8000/api'; // Backend API URL
  const WS_BASE_URL = 'ws://localhost:8000/ws'; // Backend WebSocket URL

  // Function to initialize session
  const initializeAppSession = useCallback(async () => {
    setError(null);
    setIsConnecting(true);
    console.log("Initializing session...");
    try {
      const response = await fetch(`${API_BASE_URL}/init-session`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        // Provide necessary context from exhibitor setup (hardcoded for now)
        body: JSON.stringify({
          company_name: "ABC Roasters",
          event_name: "CMPL Mumbai Expo 2025",
          company_info: "Premium coffee supplier specializing in high-quality beans.",
          products_info: "Vanilla Blend, Dark Roast Blend, Single-Origin beans, Custom Blends."
        }),
      });
      if (!response.ok) {
        const errorData = await response.text();
        throw new Error(`HTTP error! status: ${response.status} - ${errorData}`);
      }
      const data = await response.json();
      console.log("Session initialized:", data);
      setSessionId(data.session_id);
      // setCurrentStage('greeting'); // Assuming greeting is the first stage - REMOVED
      if (data.initial_message) {
        // Add initial message from backend, mark as final
        setMessages([{ role: 'assistant', content: data.initial_message, timestamp: new Date(), final: true }]);
      }
      // Connect WebSocket after getting session ID
      connectWebSocket(data.session_id);
    } catch (err) {
      console.error("Failed to initialize session:", err);
      setError(`Failed to initialize session: ${err.message}. Is the backend running?`);
      setIsConnecting(false);
    }
  }, []); // Empty dependency array means this runs once on mount

  // Function to connect WebSocket
  const connectWebSocket = (currentSessionId) => {
    if (!currentSessionId || ws.current) return;

    console.log(`Connecting WebSocket for session: ${currentSessionId}...`);
    ws.current = new WebSocket(`${WS_BASE_URL}/${currentSessionId}`);

    ws.current.onopen = () => {
      console.log("WebSocket Connected");
      setIsConnected(true);
      setIsConnecting(false);
      setError(null);
    };

    ws.current.onmessage = (event) => {
      console.log("WebSocket Message Received:", event.data);
      try {
        const messageData = JSON.parse(event.data);

        if (messageData.type === 'stream') {
           setMessages(prev => {
                const lastMsg = prev[prev.length - 1];
                if (lastMsg && lastMsg.role === 'assistant' && !lastMsg.final) {
                    const updatedMsg = { ...lastMsg, content: lastMsg.content + messageData.payload.chunk };
                    return [...prev.slice(0, -1), updatedMsg];
                } else {
                    return [...prev, { role: 'assistant', content: messageData.payload.chunk, timestamp: new Date(), final: false }];
                }
            });
        } else if (messageData.type === 'final_response') {
             setMessages(prev => {
                const lastMsg = prev[prev.length - 1];
                if (lastMsg && lastMsg.role === 'assistant' && !lastMsg.final) {
                     const finalMsg = { ...lastMsg, content: messageData.payload.response, timestamp: new Date(), final: true };
                     return [...prev.slice(0, -1), finalMsg];
                } else {
                    // Ensure we don't add duplicate final messages if connection glitches
                    if (lastMsg?.content !== messageData.payload.response || lastMsg?.role !== 'assistant') {
                         return [...prev, { role: 'assistant', content: messageData.payload.response, timestamp: new Date(), final: true }];
                    }
                    return prev; // Avoid adding duplicate
                }
            });
            // setCurrentStage(messageData.payload.current_stage); - REMOVED
            // --- Update Lead Info based on stage or specific backend signals ---
            // This is a simplified update; ideally, backend sends updated profile data
            if (messageData.payload.current_stage === 'product_discussion' || messageData.payload.current_stage === 'closing') {
                 // Example: Try to extract name/company if available in messages (very basic)
                 const humanMessages = messages.filter(m => m.role === 'human');
                 const nameGuess = humanMessages.find(m => m.content.toLowerCase().includes("i'm"))?.content.split(" ")[1]; // Highly unreliable
                 const companyGuess = humanMessages.find(m => m.content.toLowerCase().includes("from"))?.content.split("from ")[1]; // Highly unreliable
                 setLeadInfo(prev => ({
                     ...prev,
                     full_name: prev.full_name || nameGuess || prev.full_name,
                     company: prev.company || companyGuess || prev.company
                 }));
            }
            // A better approach: Backend sends updated lead_profile in the messageData payload
            // if (messageData.payload.lead_profile) {
            //    setLeadInfo(prev => ({ ...prev, ...messageData.payload.lead_profile }));
            // }

        } else if (messageData.type === 'tool_result') {
            console.log("Tool Result:", messageData.payload);
             // Handle business card scan results with confirmation step
             if (messageData.payload.name === 'scan_business_card') {
                 try {
                     const scanData = JSON.parse(messageData.payload.output);
                     if (scanData.status !== 'disabled' && scanData.status !== 'error' && !scanData.error) {
                         // Store the scan data for confirmation instead of immediately updating
                         setMessages(prev => [
                             ...prev, 
                             { 
                                 role: 'system', 
                                 content: `Business card scan successful!\n\nName: ${scanData.full_name || 'Not detected'}\nCompany: ${scanData.company || 'Not detected'}\n\nIs this information correct?`, 
                                 timestamp: new Date(),
                                 scanData: scanData, // Store the scan data with the message
                                 requiresConfirmation: true // Flag to show confirm buttons
                             }
                         ]);
                     } else {
                         // Show error message for failed scan
                         setMessages(prev => [
                             ...prev,
                             {
                                 role: 'system',
                                 content: `Business card scan failed: ${scanData.message || 'Could not read card clearly'}`,
                                 timestamp: new Date()
                             }
                         ]);
                     }
                 } catch (e) { 
                     console.error("Failed to parse scan_business_card output", e);
                     setMessages(prev => [
                         ...prev,
                         {
                             role: 'system',
                             content: `Error processing business card: ${e.message}`,
                             timestamp: new Date()
                         }
                     ]);
                 }
             }
        } else if (messageData.type === 'error') {
            setError(`WebSocket Error: ${messageData.payload.message}`);
        } else if (messageData.type === 'status' && messageData.payload.message === 'Conversation ended.') {
             console.log("Conversation ended by backend.");
             // Potentially disable input, show export options more prominently
        }
      } catch (err) {
        console.error("Failed to parse WebSocket message or update state:", err);
      }
    };

    ws.current.onerror = (event) => {
      console.error("WebSocket Error:", event);
      setError("WebSocket connection error. Check console and backend.");
      setIsConnected(false);
      setIsConnecting(false);
    };

    ws.current.onclose = (event) => {
      console.log("WebSocket Disconnected:", event.reason, event.code);
      setIsConnected(false);
      setIsConnecting(false);
      ws.current = null;
    };
  };

  // Modified version of initializeAppSession that takes a company parameter
  const initializeSessionWithCompany = useCallback(async (company) => {
    setError(null);
    setIsConnecting(true);
    console.log("Initializing session with company:", company);
    try {
      const response = await fetch(`${API_BASE_URL}/init-session`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          company_id: company.id,
          company_name: company.display_name,
          event_name: "CMPL Mumbai Expo 2025",
          company_info: company.info,
          products_info: company.products_info
        }),
      });
      if (!response.ok) {
        const errorData = await response.text();
        throw new Error(`HTTP error! status: ${response.status} - ${errorData}`);
      }
      const data = await response.json();
      console.log("Session initialized:", data);
      setSessionId(data.session_id);
      // setCurrentStage('greeting'); - REMOVED
      if (data.initial_message) {
        setMessages([{ role: 'assistant', content: data.initial_message, timestamp: new Date(), final: true }]);
      }
      connectWebSocket(data.session_id);
      
      // Update the company in header and state
      setSelectedCompany(company);
      setShowCompanySelector(false); // Hide selector after selection
    } catch (err) {
      console.error("Failed to initialize session:", err);
      setError(`Failed to initialize session: ${err.message}. Is the backend running?`);
      setIsConnecting(false);
    }
  }, [API_BASE_URL]);

  // Handle company selection 
  const handleCompanySelect = (company) => {
    initializeSessionWithCompany(company);
  };
  
  // Legacy initialization effect - empty for now, we only initialize after company selection
  useEffect(() => {
    // Cleanup WebSocket on component unmount
    return () => {
      ws.current?.close();
    };
  }, []);

  // Modify sendMessage to accept optional imageData
  const sendMessage = async (messageContent, imageData = null) => { // Added imageData parameter
    if (!sessionId) {
        setError("Session not initialized. Cannot send message.");
        return;
    }
    // Allow sending just an image with a placeholder message
    if (!messageContent.trim() && !imageData) return;

    // Don't add the placeholder message to the visual chat history if it's just for the image
    if (messageContent !== "[Business Card Image Uploaded]") {
        const userMessage = {
          role: 'human',
          content: messageContent,
          timestamp: new Date(),
        };
        setMessages(prevMessages => [...prevMessages, userMessage]);
    }
    setError(null); // Clear previous errors

    try {
        const response = await fetch(`${API_BASE_URL}/send-message`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                session_id: sessionId,
                message: messageContent, // Send the message content (could be placeholder)
                image_data: imageData // Include image data here
            }),
        });

        if (!response.ok) {
            const errorData = await response.text();
            // Add error message to chat
            setMessages(prev => [...prev, { role: 'system', content: `Error sending message/image: ${errorData}`, timestamp: new Date() }]);
            throw new Error(`HTTP error! status: ${response.status} - ${errorData}`);
        }

        // Response handling might be different now, as the backend might just acknowledge
        // and the actual AI response/tool result comes via WebSocket.
        // Let's assume the HTTP response is minimal or just confirms receipt.
        // The WebSocket handler should display the final AI response and update stage/lead info.
        const data = await response.json(); // Still parse the response
        console.log("Received response from send-message:", data);

        // --- Reinstate direct handling of AI response from HTTP POST ---
        if (data.response) {
            const assistantMessage = {
                role: 'assistant',
                content: data.response,
                timestamp: new Date(),
                final: true, // Mark as final since it's a complete response from HTTP
            };
            // Add the message, ensuring no duplicates if WS also sends it (basic check)
            setMessages(prevMessages => {
                const lastMsg = prevMessages[prevMessages.length - 1];
                if (lastMsg?.role !== 'assistant' || lastMsg?.content !== assistantMessage.content) {
                    return [...prevMessages, assistantMessage];
                }
                return prevMessages; // Avoid adding duplicate
            });
        }

        // Update the stage based on the HTTP response - REMOVED
        /* if (data.current_stage) {
            // setCurrentStage(data.current_stage); - REMOVED
        } */
    } catch (err) {
        console.error("Failed to send message or receive response:", err);
        setError(`Failed to communicate with backend: ${err.message}`);
        // Add error message to chat if not already added
        if (!messages.some(msg => msg.role === 'system' && msg.content.includes(err.message))) {
             setMessages(prev => [...prev, { role: 'system', content: `Error: ${err.message}`, timestamp: new Date() }]);
        }
    }
  };

  // Placeholder functions for OverviewPanel actions
  const handleExport = () => {
      if (!sessionId) {
          setError("No active session to export.");
          return;
      }
      console.log(`Attempting export for session: ${sessionId} via POST /api/export-lead/${sessionId}`); // More specific log
      // Call the backend export endpoint
      fetch(`${API_BASE_URL}/export-lead/${sessionId}`, { method: 'POST' })
          .then(response => {
              if (!response.ok) {
                  throw new Error(`Export failed! Status: ${response.status}`);
              }
              return response.json();
          })
          .then(data => {
              console.log("Export successful:", data);
              // Provide download or display data (e.g., copy JSON to clipboard)
              alert(`Lead data exported (check console). Profile ID: ${data.lead_profile?.id}`);
              // Update local leadInfo state with the potentially more complete exported data
              if (data.lead_profile) {
                   setLeadInfo(prev => ({ ...prev, ...data.lead_profile }));
              }
          })
          .catch(err => {
              // Log the specific error, especially useful for 404s
              console.error(`Export fetch failed for session ${sessionId}:`, err);
              setError(`Export failed: ${err.message}. Check if the backend endpoint exists and is working.`); // Updated error message
          });
  };

  const handleCreateFollowup = () => {
      console.log("Create Follow-up action triggered");
      // Implement logic to open a modal or form for creating follow-up tasks (email/meeting)
      // This might involve calling backend tools directly via specific API endpoints if needed
      alert("Follow-up creation UI not implemented yet.");
  };
  
  // Function to handle exporting the transcript
  const handleExportTranscript = () => {
      if (!sessionId) {
          setError("No active session to export transcript.");
          return;
      }
      console.log(`Requesting transcript export for session: ${sessionId}`);
      setError(null); // Clear previous errors

      fetch(`${API_BASE_URL}/export-transcript/${sessionId}`, { method: 'POST' })
          .then(response => {
              if (!response.ok) {
                  // Try to get error message from response body
                  return response.json().then(errData => {
                      throw new Error(`Transcript export failed! Status: ${response.status}. Message: ${errData.detail || 'Unknown error'}`);
                  }).catch(() => {
                      // Fallback if response is not JSON or body is empty
                      throw new Error(`Transcript export failed! Status: ${response.status}`);
                  });
              }
              return response.json();
          })
          .then(data => {
              if (data.status === 'success' && data.transcript) {
                  console.log("Transcript export successful.");
                  // Create a blob from the transcript text
                  const blob = new Blob([data.transcript], { type: 'text/plain;charset=utf-8' });
                  // Create a link element to trigger the download
                  const link = document.createElement('a');
                  link.href = URL.createObjectURL(blob);
                  link.download = `transcript_${sessionId}.txt`; // Filename for download
                  document.body.appendChild(link); // Required for Firefox
                  link.click();
                  document.body.removeChild(link); // Clean up
                  URL.revokeObjectURL(link.href); // Free up memory
                  alert("Transcript downloaded successfully!");
              } else {
                  throw new Error(data.message || "Transcript export failed: Invalid response format.");
              }
          })
          .catch(err => {
              console.error("Transcript export error:", err);
              setError(`Transcript export failed: ${err.message}`);
              alert(`Transcript export failed: ${err.message}`); // Show error to user
          });
  };

  // Function to handle switching views
  const handleViewChange = (view) => {
    setActiveView(view);
    // Reset session list selection or other states if needed when switching
  };

  // Function to handle selecting a session from the history panel
  const handleSelectHistorySession = async (conversationId) => {
    if (!conversationId) return;
    setIsHistoryModalOpen(true);
    setIsHistoryLoading(true);
    setHistoryError(null);
    setSelectedHistoryConversation(null); // Clear previous data

    try {
      const response = await fetch(`${API_BASE_URL}/conversations/${conversationId}`);
      if (!response.ok) {
        const errorData = await response.text();
        throw new Error(`HTTP error! status: ${response.status} - ${errorData}`);
      }
      const data = await response.json();
      // Assuming the endpoint returns the conversation object directly
      setSelectedHistoryConversation(data);
    } catch (err) {
      console.error("Failed to fetch conversation messages:", err);
      setHistoryError(`Failed to load messages: ${err.message}`);
    } finally {
      setIsHistoryLoading(false);
    }
  };

  // Function to close the history modal
  const handleCloseHistoryModal = () => {
    setIsHistoryModalOpen(false);
    setSelectedHistoryConversation(null);
    setHistoryError(null);
  };


  // Function to handle business card scan data from ChatArea
  const handleScanBusinessCard = (imageData) => {
    if (!sessionId) {
      setError("Session not initialized. Cannot scan card.");
      return;
    }
    if (!imageData) {
      console.warn("No image data received for scanning.");
      return;
    }
    console.log("Sending business card image via send-message endpoint...");
    setError(null); // Clear previous errors

    // Add a system message locally to indicate upload attempt
    setMessages(prev => [...prev, {
        role: 'system',
        content: 'Uploading business card image...',
        timestamp: new Date()
    }]);

    // Call the existing sendMessage function, but include the image data
    // Pass a placeholder message or null if the backend handles image-only input
    sendMessage("[Business Card Image Uploaded]", imageData);
  };
  
  // Add a new function to directly update lead info
  const confirmBusinessCardInfo = (name, company) => {
    console.log(`Direct lead info update - Name: ${name}, Company: ${company}`);
    // Directly set the state with the new values using a function to ensure we get the latest state
    setLeadInfo(prevInfo => ({
      ...prevInfo, // Keep all other fields
      full_name: name,
      company: company
    }));
    
    // Add a confirmation message
    setMessages(prev => [
      ...prev,
      {
        role: 'system',
        content: 'Lead information confirmed and updated.',
        timestamp: new Date()
      }
    ]);
  };


  return (
    <div className="app-container">
      <header className="app-header">
        <div className="header-left">
          <h1>Trade Show Lead Management</h1>
          <p>CMPL Mumbai Expo 2025</p>
        </div>
        <div className="header-right">
          <button className="theme-toggle" onClick={toggleDarkMode} title="Toggle dark mode">
            {isDarkMode ? <FaSun /> : <FaMoon />}
          </button>
          <span className={`connection-status ${isConnected ? 'connected' : 'disconnected'}`}>
            {isConnecting ? 'Connecting...' : (isConnected ? 'Connected' : 'Disconnected')}
          </span>
          {selectedCompany && (
            <>
              <button className="active-convo-btn">Active Conversation</button>
              <button className="company-btn">{selectedCompany.display_name}</button>
            </>
          )}
        </div>
      </header>

      {error && <div className="error-banner">{error}</div>}

      <main className="main-content">
        {showCompanySelector ? (
          <div className="company-selector-container">
            <CompanySelector 
              onSelectCompany={handleCompanySelect}
              apiBaseUrl={API_BASE_URL}
              eventName="CMPL Mumbai Expo 2025"
            />
          </div>
        ) : (
          <>
            <Sidebar activeView={activeView} onViewChange={handleViewChange} />
            {activeView === 'chat' ? (
              <>
                <ChatArea
                  messages={messages}
                  onSendMessage={sendMessage}
                  // stage={currentStage} - REMOVED
                  onScanBusinessCard={handleScanBusinessCard}
                  onConfirmBusinessCard={confirmBusinessCardInfo}
                  setMessages={setMessages}
                />
                <OverviewPanel
                  leadInfo={leadInfo}
                  // stage={currentStage} - REMOVED
                  onExport={handleExportTranscript} // Pass the NEW transcript export function
                  onCreateFollowup={handleCreateFollowup}
                  company={selectedCompany}
                  key={Date.now()} // Force re-render on every render to ensure updates are shown
                />
              </>
            ) : (
              <ConversationHistoryPanel
                onSelectSession={handleSelectHistorySession}
                apiBaseUrl={API_BASE_URL}
              />
            )}
          </>
        )}
      </main>

      {/* History Modal */}
      <HistoryModal
        isOpen={isHistoryModalOpen}
        onClose={handleCloseHistoryModal}
        conversation={selectedHistoryConversation}
        isLoading={isHistoryLoading}
        error={historyError}
      />
    </div>
  );
}

export default App;
