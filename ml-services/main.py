"""
CMLRE Marine Platform - ML Services
Main FastAPI application for machine learning services
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import os
from contextlib import asynccontextmanager
import logging

from services.otolith_analysis.service import OtolithAnalysisService
from services.taxonomy_classification.service import TaxonomyClassificationService
from services.molecular_analysis.service import MolecularAnalysisService
from services.data_integration.service import DataIntegrationService
from models.requests import (
    OtolithAnalysisRequest,
    TaxonomyClassificationRequest,
    MolecularAnalysisRequest,
    DataIntegrationRequest
)
from models.responses import (
    OtolithAnalysisResponse,
    TaxonomyClassificationResponse,
    MolecularAnalysisResponse,
    DataIntegrationResponse
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global service instances
otolith_service = None
taxonomy_service = None
molecular_service = None
integration_service = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize services on startup"""
    global otolith_service, taxonomy_service, molecular_service, integration_service
    
    logger.info("Initializing ML services...")
    
    try:
        # Initialize services
        otolith_service = OtolithAnalysisService()
        taxonomy_service = TaxonomyClassificationService()
        molecular_service = MolecularAnalysisService()
        integration_service = DataIntegrationService()
        
        # Load models
        await otolith_service.load_models()
        await taxonomy_service.load_models()
        await molecular_service.load_models()
        
        logger.info("All ML services initialized successfully")
        yield
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise
    finally:
        # Cleanup on shutdown
        logger.info("Shutting down ML services...")

# Create FastAPI app
app = FastAPI(
    title="CMLRE Marine Platform - ML Services",
    description="Machine learning services for marine data analysis",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "services": {
            "otolith_analysis": otolith_service is not None,
            "taxonomy_classification": taxonomy_service is not None,
            "molecular_analysis": molecular_service is not None,
            "data_integration": integration_service is not None
        }
    }

# Otolith Analysis Endpoints
@app.post("/otolith/analyze", response_model=OtolithAnalysisResponse)
async def analyze_otolith(request: OtolithAnalysisRequest):
    """Analyze otolith shape and morphometrics"""
    try:
        if otolith_service is None:
            raise HTTPException(status_code=503, detail="Otolith analysis service not available")
        
        result = await otolith_service.analyze_otolith(request)
        return result
    except Exception as e:
        logger.error(f"Error in otolith analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/otolith/classify")
async def classify_otolith_species(request: OtolithAnalysisRequest):
    """Classify species based on otolith morphology"""
    try:
        if otolith_service is None:
            raise HTTPException(status_code=503, detail="Otolith analysis service not available")
        
        result = await otolith_service.classify_species(request)
        return result
    except Exception as e:
        logger.error(f"Error in otolith classification: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Taxonomy Classification Endpoints
@app.post("/taxonomy/classify", response_model=TaxonomyClassificationResponse)
async def classify_taxonomy(request: TaxonomyClassificationRequest):
    """Classify species using morphological and molecular data"""
    try:
        if taxonomy_service is None:
            raise HTTPException(status_code=503, detail="Taxonomy classification service not available")
        
        result = await taxonomy_service.classify_species(request)
        return result
    except Exception as e:
        logger.error(f"Error in taxonomy classification: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/taxonomy/identify")
async def identify_species(request: TaxonomyClassificationRequest):
    """Identify species from images or morphological data"""
    try:
        if taxonomy_service is None:
            raise HTTPException(status_code=503, detail="Taxonomy classification service not available")
        
        result = await taxonomy_service.identify_species(request)
        return result
    except Exception as e:
        logger.error(f"Error in species identification: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Molecular Analysis Endpoints
@app.post("/molecular/analyze", response_model=MolecularAnalysisResponse)
async def analyze_molecular_data(request: MolecularAnalysisRequest):
    """Analyze molecular data including DNA sequences and eDNA"""
    try:
        if molecular_service is None:
            raise HTTPException(status_code=503, detail="Molecular analysis service not available")
        
        result = await molecular_service.analyze_sequence(request)
        return result
    except Exception as e:
        logger.error(f"Error in molecular analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/molecular/edna")
async def analyze_edna(request: MolecularAnalysisRequest):
    """Analyze environmental DNA data"""
    try:
        if molecular_service is None:
            raise HTTPException(status_code=503, detail="Molecular analysis service not available")
        
        result = await molecular_service.analyze_edna(request)
        return result
    except Exception as e:
        logger.error(f"Error in eDNA analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Data Integration Endpoints
@app.post("/integration/correlate", response_model=DataIntegrationResponse)
async def correlate_data(request: DataIntegrationRequest):
    """Perform cross-disciplinary correlation analysis"""
    try:
        if integration_service is None:
            raise HTTPException(status_code=503, detail="Data integration service not available")
        
        result = await integration_service.correlate_data(request)
        return result
    except Exception as e:
        logger.error(f"Error in data correlation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/integration/visualize")
async def create_visualization(request: DataIntegrationRequest):
    """Create integrated visualizations of marine data"""
    try:
        if integration_service is None:
            raise HTTPException(status_code=503, detail="Data integration service not available")
        
        result = await integration_service.create_visualization(request)
        return result
    except Exception as e:
        logger.error(f"Error in visualization creation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
