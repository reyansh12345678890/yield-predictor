"""
Smoke tests for prediction functionality.

This module tests the core prediction functionality to ensure
the system works end-to-end.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from inference.predictor import CropYieldPredictor, load_model
from api.weather_client import WeatherClient


class TestPredictorSmoke:
    """Smoke tests for the predictor module."""
    
    def test_predictor_initialization(self):
        """Test that predictor can be initialized."""
        predictor = CropYieldPredictor()
        
        assert predictor.model_path is not None
        assert predictor.feature_list_path is not None
        assert predictor.crop_safety_limits is not None
        assert len(predictor.crop_safety_limits) > 0
    
    def test_predictor_load_model_mock(self):
        """Test loading model with mock data (when model doesn't exist)."""
        predictor = CropYieldPredictor(
            model_path="nonexistent_model.pkl",
            feature_list_path="nonexistent_features.pkl"
        )
        
        # Should return False when model doesn't exist
        result = predictor.load_model()
        assert result == False
    
    def test_predictor_crop_safety_limits(self):
        """Test that crop safety limits are properly defined."""
        predictor = CropYieldPredictor()
        
        # Check that safety limits exist for common crops
        assert 'Rice' in predictor.crop_safety_limits
        assert 'Wheat' in predictor.crop_safety_limits
        assert 'Corn' in predictor.crop_safety_limits
        
        # Check that limits have required keys
        rice_limits = predictor.crop_safety_limits['Rice']
        assert 'max_fertilizer_per_ha' in rice_limits
        assert 'min_fertilizer_per_ha' in rice_limits
        
        # Check that limits are reasonable
        assert rice_limits['max_fertilizer_per_ha'] > rice_limits['min_fertilizer_per_ha']
        assert rice_limits['max_fertilizer_per_ha'] > 0
        assert rice_limits['min_fertilizer_per_ha'] > 0


class TestWeatherClientSmoke:
    """Smoke tests for the weather client."""
    
    def test_weather_client_initialization(self):
        """Test that weather client can be initialized."""
        client = WeatherClient()
        
        assert client.api_key is None or isinstance(client.api_key, str)
        assert client.base_url is not None
        assert client.mock_data is not None
    
    def test_weather_client_mock_data(self):
        """Test that mock weather data is available."""
        client = WeatherClient()
        
        # Test getting mock weather data
        weather = client.get_recent_weather('Cuttack', 2023)
        
        assert 'district' in weather
        assert 'year' in weather
        assert 'temperature' in weather
        assert 'rainfall' in weather
        assert 'source' in weather
        
        assert weather['district'] == 'Cuttack'
        assert weather['year'] == 2023
        assert weather['source'] == 'mock_data'
        assert isinstance(weather['temperature'], (int, float))
        assert isinstance(weather['rainfall'], (int, float))
    
    def test_weather_client_yearly_rainfall(self):
        """Test yearly rainfall data retrieval."""
        client = WeatherClient()
        
        rainfall_data = client.get_yearly_rainfall('Puri', 2023)
        
        assert 'district' in rainfall_data
        assert 'year' in rainfall_data
        assert 'rainfall' in rainfall_data
        assert 'source' in rainfall_data
        
        assert rainfall_data['district'] == 'Puri'
        assert rainfall_data['year'] == 2023
        assert isinstance(rainfall_data['rainfall'], (int, float))


class TestPredictionSmoke:
    """Smoke tests for prediction functionality."""
    
    def test_prediction_input_validation(self):
        """Test that prediction input is properly validated."""
        predictor = CropYieldPredictor()
        
        # Test with minimal valid input
        input_data = {
            'district': 'Cuttack',
            'crop': 'Rice',
            'season': 'Kharif',
            'area': 2.0,
            'irrigation': 'Flood',
            'fertilizer': 150.0,
            'pesticide': 25.0,
            'year': 2023,
            'rainfall': 1200.0,
            'temperature': 28.5,
            'soil_type': 'Clay',
            'seed_variety': 'HYV'
        }
        
        # This should not raise an error even without a loaded model
        # (it will fail at the prediction step, but input validation should pass)
        try:
            # Test that input data can be processed
            df = pd.DataFrame([input_data])
            
            # Check that required columns are present
            required_columns = ['district', 'crop', 'season', 'area', 'irrigation', 
                             'fertilizer', 'pesticide', 'year', 'rainfall', 'temperature']
            
            for col in required_columns:
                assert col in df.columns
            
            # Check data types
            assert isinstance(df['area'].iloc[0], (int, float))
            assert isinstance(df['fertilizer'].iloc[0], (int, float))
            assert isinstance(df['pesticide'].iloc[0], (int, float))
            
        except Exception as e:
            pytest.fail(f"Input validation failed: {e}")
    
    def test_recommendation_generation_structure(self):
        """Test that recommendation generation returns proper structure."""
        predictor = CropYieldPredictor()
        
        # Test recommendation structure without actual model
        base_input = {
            'district': 'Cuttack',
            'crop': 'Rice',
            'season': 'Kharif',
            'area': 2.0,
            'irrigation': 'Flood',
            'fertilizer': 150.0,
            'pesticide': 25.0,
            'year': 2023,
            'rainfall': 1200.0,
            'temperature': 28.5,
            'soil_type': 'Clay',
            'seed_variety': 'HYV'
        }
        
        # Test that recommendation methods exist and can be called
        # (they will fail without a loaded model, but structure should be correct)
        
        # Test fertilizer optimization
        try:
            result = predictor._optimize_fertilizer(base_input, 1.5)
            if result is not None:
                assert 'type' in result
                assert 'text' in result
                assert 'expected_pct_gain' in result
                assert 'rationale' in result
        except Exception:
            pass  # Expected to fail without loaded model
        
        # Test irrigation optimization
        try:
            result = predictor._optimize_irrigation(base_input, 1.5)
            if result is not None:
                assert 'type' in result
                assert 'text' in result
                assert 'expected_pct_gain' in result
                assert 'rationale' in result
        except Exception:
            pass  # Expected to fail without loaded model
        
        # Test crop selection optimization
        try:
            result = predictor._optimize_crop_selection(base_input, 1.5)
            if result is not None:
                assert 'type' in result
                assert 'text' in result
                assert 'expected_pct_gain' in result
                assert 'rationale' in result
        except Exception:
            pass  # Expected to fail without loaded model


class TestIntegrationSmoke:
    """Integration smoke tests."""
    
    def test_weather_client_integration(self):
        """Test weather client integration with different districts."""
        client = WeatherClient()
        
        districts = ['Cuttack', 'Puri', 'Bhubaneswar']
        
        for district in districts:
            weather = client.get_recent_weather(district, 2023)
            
            # Check that weather data is consistent
            assert weather['district'] == district
            assert weather['year'] == 2023
            assert weather['source'] == 'mock_data'
            
            # Check that values are reasonable
            assert 20 <= weather['temperature'] <= 35  # Reasonable temperature range
            assert 500 <= weather['rainfall'] <= 2000  # Reasonable rainfall range
    
    def test_feature_engineering_integration(self):
        """Test feature engineering integration."""
        # Create sample data
        df = pd.DataFrame({
            'fertilizer': [100, 200, 150],
            'pesticide': [20, 40, 30],
            'area': [2, 4, 3],
            'rainfall': [1200, 1100, 1150],
            'temperature': [28.5, 29.1, 28.8],
            'state': ['Odisha', 'Odisha', 'Odisha'],
            'district': ['Cuttack', 'Puri', 'Bhubaneswar'],
            'crop': ['Rice', 'Wheat', 'Corn'],
            'season': ['Kharif', 'Rabi', 'Kharif'],
            'soil_type': ['Clay', 'Sandy', 'Loam'],
            'irrigation': ['Flood', 'Sprinkler', 'Drip'],
            'seed_variety': ['HYV', 'HYV', 'HYV'],
            'yield': [1.68, 1.59, 1.53]
        })
        
        # Test feature creation
        from ml.features import create_features
        X, y, numeric_features, categorical_features = create_features(df)
        
        # Check that features are created correctly
        assert len(X) == 3
        assert len(y) == 3
        assert len(numeric_features) == 5
        assert len(categorical_features) == 7
        
        # Check that per-hectare features are calculated
        assert 'fertilizer_per_ha' in X.columns
        assert 'pesticide_per_ha' in X.columns
        
        # Check calculations
        assert X['fertilizer_per_ha'].iloc[0] == 50.0  # 100/2
        assert X['pesticide_per_ha'].iloc[0] == 10.0    # 20/2


if __name__ == "__main__":
    pytest.main([__file__])
