import React, { useState, useEffect } from 'react';
import axios from 'axios';
// Removed react-spring import
import './App.css';

function App() {
  const [activePage, setActivePage] = useState('home');
  const [spendingData, setSpendingData] = useState({
    Fresh: '',
    Milk: '',
    Grocery: '',
    Frozen: '',
    Detergents_Paper: '',
    Delicassen: ''
  });
  const [segmentResult, setSegmentResult] = useState(null);
  const [personas, setPersonas] = useState([]);
  const [pcaData, setPcaData] = useState([]);
  const [elbowData, setElbowData] = useState([]);
  const [loading, setLoading] = useState(false);

  // Fetch personas on component mount
  useEffect(() => {
    fetchPersonas();
    fetchPcaData();
    fetchElbowData();
  }, []);

  const fetchPersonas = async () => {
    try {
      // In a real app, this would be an API call
      // const response = await axios.get('http://localhost:8000/clusters');
      // setPersonas(response.data);
      
      // For now, we'll use mock data
      const mockPersonas = [
        {
          cluster_id: 0,
          size: 489,
          dominant_category: "Fresh",
          weakest_category: "Delicassen",
          behavioral_tag: "Fresh Produce Specialists",
          persona_summary: "Fresh Produce Specialists who primarily purchase fresh products. They show minimal interest in delicassen items. These customers are high-volume fresh produce buyers. Target with premium quality offerings and seasonal promotions.",
          campaign_recommendation: "Promote organic and specialty fresh produce lines with volume discounts."
        },
        {
          cluster_id: 1,
          size: 14,
          dominant_category: "Fresh",
          weakest_category: "Frozen",
          behavioral_tag: "Fresh Produce Specialists",
          persona_summary: "Fresh Produce Specialists who primarily purchase fresh products. They show minimal interest in frozen items. These customers focus heavily on grocery items. They represent stable, recurring revenue opportunities.",
          campaign_recommendation: "Introduce loyalty programs and bundled grocery packages."
        },
        {
          cluster_id: 2,
          size: 6,
          dominant_category: "Fresh",
          weakest_category: "Detergents_Paper",
          behavioral_tag: "Fresh Produce Specialists",
          persona_summary: "Fresh Produce Specialists who primarily purchase fresh products. They show minimal interest in detergents_paper items. These customers have diverse purchasing patterns with balanced spending across categories.",
          campaign_recommendation: "Offer cross-category promotions and personalized recommendations."
        }
      ];
      setPersonas(mockPersonas);
    } catch (error) {
      console.error('Error fetching personas:', error);
    }
  };

  const fetchPcaData = async () => {
    try {
      // In a real app, this would be an API call
      // const response = await axios.get('http://localhost:8000/pca');
      // setPcaData(response.data);
      
      // For now, we'll use mock data
      const mockPcaData = [
        { PC1: 8.27, PC2: -0.75, Cluster: 1 },
        { PC1: 12.72, PC2: -3.22, Cluster: 1 },
        { PC1: 17.50, PC2: 2.74, Cluster: 1 },
        { PC1: 0.52, PC2: 9.70, Cluster: 2 },
        { PC1: 7.27, PC2: 13.42, Cluster: 2 },
        { PC1: 9.51, PC2: -0.78, Cluster: 1 },
        { PC1: 7.03, PC2: 0.16, Cluster: 1 },
        { PC1: 0.31, PC2: 6.97, Cluster: 2 },
        { PC1: -0.95, PC2: -0.42, Cluster: 0 },
        { PC1: -1.66, PC2: 0.92, Cluster: 0 }
      ];
      setPcaData(mockPcaData);
    } catch (error) {
      console.error('Error fetching PCA data:', error);
    }
  };

  const fetchElbowData = async () => {
    try {
      // In a real app, this would be an API call
      // const response = await axios.get('http://localhost:8000/elbow');
      // setElbowData(response.data);
      
      // For now, we'll use mock data
      const mockElbowData = {
        k_values: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        inertias: [3054, 1766, 1260, 1042, 850, 629, 507, 399, 326, 255]
      };
      setElbowData(mockElbowData);
    } catch (error) {
      console.error('Error fetching elbow data:', error);
    }
  };

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setSpendingData({
      ...spendingData,
      [name]: value
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    
    try {
      // In a real app, this would be an API call
      // const response = await axios.post('http://localhost:8000/segment', spendingData);
      // setSegmentResult(response.data);
      
      // For now, we'll use mock data
      const mockResult = {
        cluster: Math.floor(Math.random() * 3),
        persona: personas[Math.floor(Math.random() * personas.length)],
        dominant_category: "Fresh",
        recommended_campaign: "Promote organic and specialty fresh produce lines with volume discounts."
      };
      
      setSegmentResult(mockResult);
      setActivePage('result');
    } catch (error) {
      console.error('Error segmenting customer:', error);
    } finally {
      setLoading(false);
    }
  };

  const renderHome = () => (
    // Removed animated.div and pageTransition
    <div className="page fade-in">
      <div className="hero">
        <h1>Wholesale Customer Segmentation</h1>
        <p>Discover valuable customer insights through advanced machine learning segmentation</p>
        <button 
          className="cta-button" 
          onClick={() => setActivePage('form')}
        >
          Segment a Customer
        </button>
      </div>
      
      <div className="features">
        <div className="feature-card">
          <h3>AI-Powered Segmentation</h3>
          <p>Using k-Means clustering to identify distinct customer groups</p>
        </div>
        <div className="feature-card">
          <h3>Actionable Insights</h3>
          <p>Get detailed personas with marketing recommendations</p>
        </div>
        <div className="feature-card">
          <h3>Data Visualization</h3>
          <p>Interactive charts to explore customer segments</p>
        </div>
      </div>
    </div>
  );

  const renderForm = () => (
    // Removed animated.div and pageTransition
    <div className="page fade-in">
      <h2>Customer Segmentation Form</h2>
      <p>Enter the customer's annual spending in each category to determine their segment</p>
      
      <form onSubmit={handleSubmit} className="segmentation-form">
        <div className="form-group">
          <label htmlFor="Fresh">Fresh Products ($)</label>
          <input
            type="number"
            id="Fresh"
            name="Fresh"
            value={spendingData.Fresh}
            onChange={handleInputChange}
            required
          />
        </div>
        
        <div className="form-group">
          <label htmlFor="Milk">Milk Products ($)</label>
          <input
            type="number"
            id="Milk"
            name="Milk"
            value={spendingData.Milk}
            onChange={handleInputChange}
            required
          />
        </div>
        
        <div className="form-group">
          <label htmlFor="Grocery">Grocery ($)</label>
          <input
            type="number"
            id="Grocery"
            name="Grocery"
            value={spendingData.Grocery}
            onChange={handleInputChange}
            required
          />
        </div>
        
        <div className="form-group">
          <label htmlFor="Frozen">Frozen Products ($)</label>
          <input
            type="number"
            id="Frozen"
            name="Frozen"
            value={spendingData.Frozen}
            onChange={handleInputChange}
            required
          />
        </div>
        
        <div className="form-group">
          <label htmlFor="Detergents_Paper">Detergents & Paper ($)</label>
          <input
            type="number"
            id="Detergents_Paper"
            name="Detergents_Paper"
            value={spendingData.Detergents_Paper}
            onChange={handleInputChange}
            required
          />
        </div>
        
        <div className="form-group">
          <label htmlFor="Delicassen">Delicassen ($)</label>
          <input
            type="number"
            id="Delicassen"
            name="Delicassen"
            value={spendingData.Delicassen}
            onChange={handleInputChange}
            required
          />
        </div>
        
        <button type="submit" className="submit-button" disabled={loading}>
          {loading ? 'Segmenting...' : 'Segment Customer'}
        </button>
      </form>
      
      <button onClick={() => setActivePage('home')} className="back-button">
        ← Back to Home
      </button>
    </div>
  );

  const renderResult = () => (
    // Removed animated.div and pageTransition
    <div className="page fade-in">
      <h2>Segmentation Result</h2>
      
      {segmentResult && (
        <div className="result-card">
          <h3>Customer Segment: Cluster {segmentResult.cluster}</h3>
          <p><strong>Persona:</strong> {segmentResult.persona.behavioral_tag}</p>
          <p><strong>Dominant Category:</strong> {segmentResult.persona.dominant_category}</p>
          <p><strong>Weakest Category:</strong> {segmentResult.persona.weakest_category}</p>
          <p><strong>Persona Summary:</strong> {segmentResult.persona.persona_summary}</p>
          <p><strong>Recommended Campaign:</strong> {segmentResult.recommended_campaign}</p>
        </div>
      )}
      
      <button onClick={() => setActivePage('form')} className="back-button">
        ← Segment Another Customer
      </button>
    </div>
  );

  const renderPersonas = () => (
    // Removed animated.div and pageTransition
    <div className="page fade-in">
      <h2>Customer Personas</h2>
      <p>Detailed profiles of each customer segment</p>
      
      <div className="personas-container">
        {personas.map((persona) => (
          <div key={persona.cluster_id} className="persona-card">
            <h3>Cluster {persona.cluster_id}: {persona.behavioral_tag}</h3>
            <p><strong>Size:</strong> {persona.size} customers</p>
            <p><strong>Dominant Category:</strong> {persona.dominant_category}</p>
            <p><strong>Weakest Category:</strong> {persona.weakest_category}</p>
            <p><strong>Summary:</strong> {persona.persona_summary}</p>
            <p><strong>Marketing Recommendation:</strong> {persona.campaign_recommendation}</p>
          </div>
        ))}
      </div>
      
      <button onClick={() => setActivePage('home')} className="back-button">
        ← Back to Home
      </button>
    </div>
  );

  const renderVisualization = () => (
    // Removed animated.div and pageTransition
    <div className="page fade-in">
      <h2>Customer Segmentation Visualization</h2>
      
      <div className="visualization-section">
        <h3>PCA Visualization</h3>
        <div className="pca-chart">
          <p>PCA visualization would be displayed here</p>
          <p>X-axis: PC1 (53.7% variance)</p>
          <p>Y-axis: PC2 (22.1% variance)</p>
        </div>
      </div>
      
      <div className="visualization-section">
        <h3>Elbow Method</h3>
        <div className="elbow-chart">
          <p>Elbow chart would be displayed here</p>
          <p>Optimal number of clusters: 3</p>
        </div>
      </div>
      
      <button onClick={() => setActivePage('home')} className="back-button">
        ← Back to Home
      </button>
    </div>
  );

  return (
    <div className="App">
      <header className="App-header">
        <h1 className="app-title">Wholesale Customer Segmentation</h1>
        <nav className="navigation">
          <button 
            className={activePage === 'home' ? 'nav-button active' : 'nav-button'}
            onClick={() => setActivePage('home')}
          >
            Home
          </button>
          <button 
            className={activePage === 'form' ? 'nav-button active' : 'nav-button'}
            onClick={() => setActivePage('form')}
          >
            Segment Customer
          </button>
          <button 
            className={activePage === 'personas' ? 'nav-button active' : 'nav-button'}
            onClick={() => setActivePage('personas')}
          >
            Personas
          </button>
          <button 
            className={activePage === 'visualization' ? 'nav-button active' : 'nav-button'}
            onClick={() => setActivePage('visualization')}
          >
            Visualization
          </button>
        </nav>
      </header>
      
      <main className="App-main">
        {activePage === 'home' && renderHome()}
        {activePage === 'form' && renderForm()}
        {activePage === 'result' && renderResult()}
        {activePage === 'personas' && renderPersonas()}
        {activePage === 'visualization' && renderVisualization()}
      </main>
      
      <footer className="App-footer">
        <p>Wholesale Customer Segmentation Platform © 2023</p>
      </footer>
    </div>
  );
}

export default App;