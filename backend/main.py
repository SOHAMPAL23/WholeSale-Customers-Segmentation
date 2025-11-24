from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import pandas as pd
import numpy as np
import pickle
import json
import os

# Create FastAPI app
app = FastAPI(title="Wholesale Customer Segmentation API",
              description="API for customer segmentation using k-Means clustering",
              version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get the absolute path to the ml directory
current_dir = os.path.dirname(os.path.abspath(__file__))
ml_dir = os.path.join(current_dir, "..", "ml")

# Load models and data
try:
    # Load the trained models (adjust path to go up one directory to find ml folder)
    with open(os.path.join(ml_dir, 'kmeans.pkl'), 'rb') as f:
        kmeans_model = pickle.load(f)
    
    with open(os.path.join(ml_dir, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    
    with open(os.path.join(ml_dir, 'imputer.pkl'), 'rb') as f:
        imputer = pickle.load(f)
    
    with open(os.path.join(ml_dir, 'pca.pkl'), 'rb') as f:
        pca = pickle.load(f)
    
    # Load data
    processed_data = pd.read_csv(os.path.join(ml_dir, 'processed_data.csv'))
    pca_data = pd.read_csv(os.path.join(ml_dir, 'pca_data.csv'))
    
    # Load personas
    with open(os.path.join(ml_dir, 'personas.json'), 'r') as f:
        personas = json.load(f)
    
    # Load elbow data
    with open(os.path.join(ml_dir, 'elbow_data.json'), 'r') as f:
        elbow_data = json.load(f)
        
    print("Models and data loaded successfully!")
    
except Exception as e:
    print(f"Error loading models or data: {e}")
    kmeans_model = None
    scaler = None
    imputer = None
    pca = None
    processed_data = None
    pca_data = None
    personas = None
    elbow_data = None

# Pydantic models for request/response
class SpendingData(BaseModel):
    Fresh: float
    Milk: float
    Grocery: float
    Frozen: float
    Detergents_Paper: float
    Delicassen: float

class ClusterResult(BaseModel):
    cluster: int
    persona: Dict[str, Any]
    dominant_category: str
    recommended_campaign: str

class PersonaSummary(BaseModel):
    cluster_id: int
    size: int
    dominant_category: str
    weakest_category: str
    behavioral_tag: str
    persona_summary: str
    campaign_recommendation: str

class ElbowData(BaseModel):
    k_values: List[int]
    inertias: List[float]

class PCAPoint(BaseModel):
    PC1: float
    PC2: float
    Cluster: int

# API Endpoints
@app.get("/")
def read_root():
    return {"message": "Wholesale Customer Segmentation API", "version": "1.0.0"}

@app.post("/segment", response_model=ClusterResult)
def segment_customer(spending: SpendingData):
    """
    Accept spending values and predict cluster + persona
    """
    if kmeans_model is None or scaler is None or imputer is None:
        raise HTTPException(status_code=500, detail="Models not loaded")
    
    try:
        # Convert input to array
        spending_array = np.array([[
            spending.Fresh,
            spending.Milk,
            spending.Grocery,
            spending.Frozen,
            spending.Detergents_Paper,
            spending.Delicassen
        ]])
        
        # Handle missing values
        spending_imputed = imputer.transform(spending_array)
        
        # Scale the data
        spending_scaled = scaler.transform(spending_imputed)
        
        # Predict cluster
        cluster = int(kmeans_model.predict(spending_scaled)[0])
        
        # Get persona for this cluster
        if str(cluster) in personas:
            persona = personas[str(cluster)]
            return ClusterResult(
                cluster=cluster,
                persona=persona,
                dominant_category=persona["dominant_category"],
                recommended_campaign=persona["campaign_recommendation"]
            )
        else:
            raise HTTPException(status_code=404, detail="Persona not found for cluster")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.get("/clusters", response_model=List[PersonaSummary])
def get_cluster_personas():
    """
    Return cluster persona summaries
    """
    if personas is None:
        raise HTTPException(status_code=500, detail="Personas not loaded")
    
    try:
        result = []
        for cluster_id, persona_data in personas.items():
            persona_summary = PersonaSummary(
                cluster_id=persona_data["cluster_id"],
                size=persona_data["size"],
                dominant_category=persona_data["dominant_category"],
                weakest_category=persona_data["weakest_category"],
                behavioral_tag=persona_data["behavioral_tag"],
                persona_summary=persona_data["persona_summary"],
                campaign_recommendation=persona_data["campaign_recommendation"]
            )
            result.append(persona_summary)
        
        # Sort by cluster_id
        result.sort(key=lambda x: x.cluster_id)
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.get("/pca", response_model=List[PCAPoint])
def get_pca_data():
    """
    Return PCA-transformed 2D scatter data for all customers
    """
    if pca_data is None:
        raise HTTPException(status_code=500, detail="PCA data not loaded")
    
    try:
        result = []
        for _, row in pca_data.iterrows():
            pca_point = PCAPoint(
                PC1=float(row['PC1']),
                PC2=float(row['PC2']),
                Cluster=int(row['Cluster'])
            )
            result.append(pca_point)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.get("/elbow", response_model=ElbowData)
def get_elbow_data():
    """
    Return inertia values for elbow chart
    """
    if elbow_data is None:
        raise HTTPException(status_code=500, detail="Elbow data not loaded")
    
    try:
        return ElbowData(
            k_values=elbow_data["k_values"],
            inertias=elbow_data["inertias"]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

# Run the API
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)