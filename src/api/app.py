"""
FastAPI application for crop yield prediction and optimization.

This module provides REST API endpoints for predicting crop yield and
generating optimization recommendations.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
import pandas as pd
import json
import sys
from pathlib import Path

# Add project root and src to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.extend([str(project_root), str(project_root / 'src')])

from src.inference.predictor import CropYieldPredictor
from src.api.weather_client import WeatherClient


# Initialize FastAPI app
app = FastAPI(
    title="AI-Powered Crop Yield Prediction API",
    description="API for predicting crop yield and generating optimization recommendations",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize predictor and weather client
predictor = None
weather_client = WeatherClient()


# Pydantic models for request/response
class PredictionRequest(BaseModel):
    """Request model for single prediction."""
    state: str = Field("Odisha", description="State name")
    district: str = Field(..., description="District name")
    crop: str = Field(..., description="Crop type")
    season: str = Field(..., description="Season (Kharif/Rabi/Annual)")
    area: float = Field(..., gt=0, description="Area in hectares")
    irrigation: str = Field(..., description="Irrigation method")
    fertilizer: float = Field(..., ge=0, description="Fertilizer amount in kg")
    pesticide: float = Field(..., ge=0, description="Pesticide amount in kg")
    year: Optional[int] = Field(2023, description="Year (optional)")
    rainfall: Optional[float] = Field(None, description="Rainfall in mm (optional)")
    temperature: Optional[float] = Field(None, description="Temperature in °C (optional)")
    soil_type: Optional[str] = Field("Clay", description="Soil type (optional)")
    seed_variety: Optional[str] = Field("HYV", description="Seed variety (optional)")


class PredictionResponse(BaseModel):
    """Response model for prediction."""
    predicted_yield: float = Field(..., description="Predicted yield in t/ha")
    confidence_low: float = Field(..., description="Lower confidence bound")
    confidence_high: float = Field(..., description="Upper confidence bound")
    confidence_range: float = Field(..., description="Confidence range")
    recommendations: List[Dict[str, Any]] = Field(..., description="Optimization recommendations")
    weather_data: Dict[str, Any] = Field(..., description="Weather data used")


class BatchPredictionResponse(BaseModel):
    """Response model for batch prediction."""
    predictions: List[Dict[str, Any]] = Field(..., description="List of predictions")
    summary: Dict[str, Any] = Field(..., description="Summary statistics")


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize the predictor on startup."""
    global predictor
    try:
        predictor = CropYieldPredictor()
        if not predictor.load_model():
            raise Exception("Failed to load model")
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        # In production, you might want to fail fast here
        # For demo purposes, we'll continue without the model


# Health check endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "AI-Powered Crop Yield Prediction API",
        "version": "1.0.0",
        "status": "running",
        "model_loaded": predictor is not None
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": predictor is not None,
        "weather_client": "available"
    }


# Single prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict_yield(request: PredictionRequest):
    """
    Predict crop yield for given input parameters.
    
    If weather data (rainfall, temperature) is not provided,
    it will be fetched from weather API or mock data.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert request to dictionary
        input_data = request.dict()
        
        # Get weather data if not provided
        weather_data = {}
        if request.rainfall is None or request.temperature is None:
            weather = weather_client.get_recent_weather(request.district, request.year)
            input_data['rainfall'] = weather['rainfall']
            input_data['temperature'] = weather['temperature']
            weather_data = weather
        else:
            weather_data = {
                'district': request.district,
                'year': request.year,
                'rainfall': request.rainfall,
                'temperature': request.temperature,
                'source': 'user_provided'
            }
        
        # Fill missing optional fields with defaults
        if input_data.get('soil_type') is None:
            input_data['soil_type'] = 'Clay'
        if input_data.get('seed_variety') is None:
            input_data['seed_variety'] = 'HYV'
        
        # Make prediction
        prediction_result = predictor.predict_yield(input_data)
        
        # Generate recommendations
        recommendations = predictor.generate_recommendations(input_data)
        
        return PredictionResponse(
            predicted_yield=prediction_result['predicted_yield'],
            confidence_low=prediction_result['confidence_low'],
            confidence_high=prediction_result['confidence_high'],
            confidence_range=prediction_result['confidence_range'],
            recommendations=recommendations,
            weather_data=weather_data
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")


# Batch prediction endpoint
@app.post("/batch_predict", response_model=BatchPredictionResponse)
async def batch_predict(file: UploadFile = File(...)):
    """
    Predict crop yield for multiple records from CSV file.
    
    CSV should contain columns: district, crop, season, area, irrigation,
    fertilizer, pesticide, year (optional), rainfall (optional), temperature (optional)
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Read CSV file
        contents = await file.read()
        df = pd.read_csv(pd.io.common.StringIO(contents.decode('utf-8')))
        
        # Validate required columns
        required_columns = ['district', 'crop', 'season', 'area', 'irrigation', 'fertilizer', 'pesticide']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise HTTPException(
                status_code=400, 
                detail=f"Missing required columns: {missing_columns}"
            )
        
        predictions = []
        
        # Process each row
        for index, row in df.iterrows():
            try:
                # Convert row to dictionary
                input_data = row.to_dict()
                
                # Fill missing optional fields
                if 'year' not in input_data or pd.isna(input_data['year']):
                    input_data['year'] = 2023
                if 'soil_type' not in input_data or pd.isna(input_data['soil_type']):
                    input_data['soil_type'] = 'Clay'
                if 'seed_variety' not in input_data or pd.isna(input_data['seed_variety']):
                    input_data['seed_variety'] = 'HYV'
                
                # Get weather data if not provided
                if 'rainfall' not in input_data or pd.isna(input_data['rainfall']):
                    weather = weather_client.get_recent_weather(input_data['district'], input_data['year'])
                    input_data['rainfall'] = weather['rainfall']
                if 'temperature' not in input_data or pd.isna(input_data['temperature']):
                    weather = weather_client.get_recent_weather(input_data['district'], input_data['year'])
                    input_data['temperature'] = weather['temperature']
                
                # Make prediction
                prediction_result = predictor.predict_yield(input_data)
                
                # Generate recommendations
                recommendations = predictor.generate_recommendations(input_data)
                
                predictions.append({
                    'row_index': index,
                    'input_data': input_data,
                    'predicted_yield': prediction_result['predicted_yield'],
                    'confidence_low': prediction_result['confidence_low'],
                    'confidence_high': prediction_result['confidence_high'],
                    'recommendations': recommendations
                })
                
            except Exception as e:
                predictions.append({
                    'row_index': index,
                    'error': str(e),
                    'input_data': row.to_dict()
                })
        
        # Calculate summary statistics
        successful_predictions = [p for p in predictions if 'error' not in p]
        if successful_predictions:
            yields = [p['predicted_yield'] for p in successful_predictions]
            summary = {
                'total_records': len(df),
                'successful_predictions': len(successful_predictions),
                'failed_predictions': len(predictions) - len(successful_predictions),
                'average_yield': sum(yields) / len(yields),
                'min_yield': min(yields),
                'max_yield': max(yields)
            }
        else:
            summary = {
                'total_records': len(df),
                'successful_predictions': 0,
                'failed_predictions': len(predictions),
                'error': 'No successful predictions'
            }
        
        return BatchPredictionResponse(
            predictions=predictions,
            summary=summary
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Batch prediction failed: {str(e)}")


# Weather data endpoint
@app.get("/weather/{district}")
async def get_weather(district: str, year: int = 2023):
    """
    Get weather data for a specific district.
    """
    try:
        weather_data = weather_client.get_recent_weather(district, year)
        return weather_data
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Weather data fetch failed: {str(e)}")


# Model info endpoint
@app.get("/model/info")
async def get_model_info():
    """
    Get information about the loaded model.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        return {
            "model_path": predictor.model_path,
            "feature_list_path": predictor.feature_list_path,
            "feature_info": predictor.feature_info,
            "crop_safety_limits": predictor.crop_safety_limits
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


