import React from 'react';
import './OverviewPanel.css';

const OverviewPanel = ({ leadInfo, onExport, onCreateFollowup, company }) => { // Removed stage prop

  // Placeholder handlers - implement actual logic later
  const handleExport = () => {
    console.log("Export Transcript clicked");
    if (onExport) onExport();
  };

  const handleCreateFollowup = () => {
    console.log("Create Follow-up clicked");
    if (onCreateFollowup) onCreateFollowup();
  };

  // handleSuggestionClick and suggestedQuestions removed

  return (
    <div className="overview-panel-component"> {/* Use specific class name */}
      <h3>Conversation Overview</h3>

      <div className="lead-summary-card">
        <div className="lead-details">
            <p className="lead-name" id="lead-full-name">{leadInfo?.full_name || 'Unknown Visitor'}</p>
            <p className="lead-company" id="lead-company">{leadInfo?.company || 'Unknown Company'}</p>
            <p className="start-time">Started at: {leadInfo?.startTime || new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</p>
            {/* Debug info for troubleshooting */}
            <p className="debug-info" style={{fontSize: '10px', color: '#aaa'}}>
              Lead data: {JSON.stringify(leadInfo || {})}
            </p>
        </div>
        {/* {stage && <span className="stage-badge-overview">Stage: {stage}</span>} - REMOVED */}
      </div>
      
      {company && (
        <div className="company-info-section">
          <h4>Representing: {company.display_name}</h4>
          <div className="company-type">
            <span className="company-type-badge-overview">
              {company.company_type === "service" ? "Service Provider" : "Product Company"}
            </span>
          </div>
          {company.product_varieties && company.product_varieties.length > 0 && (
            <div className="product-varieties-card">
              <h5>{company.company_type === "service" ? "Service Offerings" : "Product Varieties"}</h5>
              <ul>
                {company.product_varieties.map((product, index) => (
                  <li key={index}>{product}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}

      {leadInfo?.collected_fields && Object.keys(leadInfo.collected_fields).length > 0 && (
        <div className="collected-fields-section">
          <h4>Collected Information</h4>
          <div className="collected-fields-list">
            {Object.entries(leadInfo.collected_fields).map(([key, value]) => {
              // Skip displaying empty values
              if (!value) return null;
              
              // Format the field key for display
              let displayKey = key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
              
              // Use field labels from company config if available
              if (company?.field_labels && company.field_labels[key]) {
                displayKey = company.field_labels[key];
              }
              
              return (
                <div key={key} className="collected-field-item">
                  <span className="field-label">{displayKey}:</span>
                  <span className="field-value">{value}</span>
                </div>
              );
            })}
          </div>
        </div>
      )}

      <div className="quick-actions-section">
        <h4>Quick Actions</h4>
        <button onClick={handleExport}>Export Transcript</button>
        <button onClick={handleCreateFollowup}>Create Follow-up</button>
      </div>

      {/* Suggested questions section removed */}

      {/* Add more sections later if needed, e.g., Extracted Insights */}

    </div>
  );
};

export default OverviewPanel;
