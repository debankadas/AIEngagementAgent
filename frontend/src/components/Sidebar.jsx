import React from 'react';
import './Sidebar.css'; // We'll create this for styling

// Placeholder icons - replace with actual icons (e.g., from react-icons) later
const ChatIcon = () => <span>💬</span>;
const PersonIcon = () => <span>👤</span>;
const CalendarIcon = () => <span>📅</span>;
const HistoryIcon = () => <span>🕒</span>;
const UploadIcon = () => <span>⬆️</span>;

const Sidebar = ({ activeView, onViewChange }) => { // Accept props
  return (
    <nav className="sidebar">
      {/* Chat Button */}
      <button
        className={`sidebar-button ${activeView === 'chat' ? 'active' : ''}`}
        title="Active Conversation"
        onClick={() => onViewChange('chat')} // Switch to chat view
      >
        <ChatIcon />
      </button>
      {/* Lead Profile Button (Placeholder) */}
      <button className="sidebar-button" title="Lead Profile" onClick={() => alert('Lead Profile feature not implemented yet.')}>
        <PersonIcon />
      </button>
      <button className="sidebar-button" title="Schedule Meeting">
        <CalendarIcon />
      </button>
      {/* History Button */}
      <button
        className={`sidebar-button ${activeView === 'history' ? 'active' : ''}`}
        title="Conversation History"
        onClick={() => onViewChange('history')} // Switch to history view
      >
        <HistoryIcon />
      </button>
      <div className="sidebar-spacer"></div> {/* Pushes upload to bottom */}
      <button className="sidebar-button sidebar-bottom" title="Upload/Export">
        <UploadIcon />
      </button>
    </nav>
  );
};

export default Sidebar;
