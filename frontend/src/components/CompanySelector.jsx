import React, { useState, useEffect } from 'react';
import './CompanySelector.css'; // We'll create this for styling

const CompanySelector = ({ onSelectCompany, apiBaseUrl, eventName }) => {
  const [companies, setCompanies] = useState([]);
  const [selectedCompany, setSelectedCompany] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  // Load companies from the backend
  useEffect(() => {
    const fetchCompanies = async () => {
      setIsLoading(true);
      setError(null);
      try {
        const response = await fetch(`${apiBaseUrl}/companies`);
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        setCompanies(data.companies || []);
        
        // Set the first company as default if available
        if (data.companies && data.companies.length > 0) {
          setSelectedCompany(data.companies[0]);
        }
      } catch (e) {
        console.error("Failed to fetch companies:", e);
        setError(`Failed to load companies: ${e.message}`);
      } finally {
        setIsLoading(false);
      }
    };

    fetchCompanies();
  }, [apiBaseUrl]);

  // Handle company selection
  const handleCompanySelect = (company) => {
    setSelectedCompany(company);
  };

  // Handle start session button click
  const handleStartSession = () => {
    if (selectedCompany && onSelectCompany) {
      onSelectCompany(selectedCompany);
    }
  };

  if (isLoading) {
    return <div className="company-selector loading">Loading available companies...</div>;
  }

  if (error) {
    return (
      <div className="company-selector error">
        <p>{error}</p>
        <button onClick={() => window.location.reload()}>Retry</button>
      </div>
    );
  }

  return (
    <div className="company-selector">
      <h2>Select a Company to Represent</h2>
      <h3>{eventName || 'Trade Show'}</h3>
      
      <div className="companies-grid">
        {companies.map((company) => (
          <div
            key={company.id}
            className={`company-card ${selectedCompany?.id === company.id ? 'selected' : ''}`}
            onClick={() => handleCompanySelect(company)}
          >
            <h3>{company.display_name}</h3>
            <div className="company-type-badge">
              {company.company_type === "service" ? "Service Provider" : "Product Company"}
            </div>
            <p className="company-info">{company.info.substring(0, 100)}{company.info.length > 100 ? '...' : ''}</p>
            <div className="product-varieties">
              <strong>{company.company_type === "service" ? "Services:" : "Products:"}</strong> 
              <p>{company.product_varieties.join(', ')}</p>
            </div>
          </div>
        ))}
      </div>

      {companies.length === 0 && (
        <div className="no-companies">
          <p>No companies available. Please check your configuration.</p>
        </div>
      )}

      <div className="action-buttons">
        <button 
          onClick={handleStartSession} 
          disabled={!selectedCompany}
          className="start-button"
        >
          {selectedCompany ? `Represent ${selectedCompany.display_name}` : 'Select a Company'}
        </button>
      </div>
    </div>
  );
};

export default CompanySelector;
