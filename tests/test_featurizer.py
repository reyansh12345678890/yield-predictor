"""
Unit tests for feature engineering module.

This module tests the core functionality of the feature engineering functions.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from ml.features import (
    normalize_per_ha,
    create_features,
    fill_missing_values,
    validate_input_data,
    get_feature_summary
)


class TestNormalizePerHa:
    """Test cases for normalize_per_ha function."""
    
    def test_normalize_per_ha_basic(self):
        """Test basic normalization functionality."""
        # Create test data
        df = pd.DataFrame({
            'fertilizer': [100, 200, 150],
            'pesticide': [20, 40, 30],
            'area': [2, 4, 3]
        })
        
        result = normalize_per_ha(df)
        
        # Check that new columns are added
        assert 'fertilizer_per_ha' in result.columns
        assert 'pesticide_per_ha' in result.columns
        
        # Check calculations
        expected_fertilizer_per_ha = [50, 50, 50]  # 100/2, 200/4, 150/3
        expected_pesticide_per_ha = [10, 10, 10]   # 20/2, 40/4, 30/3
        
        assert result['fertilizer_per_ha'].tolist() == expected_fertilizer_per_ha
        assert result['pesticide_per_ha'].tolist() == expected_pesticide_per_ha
    
    def test_normalize_per_ha_zero_area(self):
        """Test handling of zero area (should raise error)."""
        df = pd.DataFrame({
            'fertilizer': [100],
            'pesticide': [20],
            'area': [0]  # Zero area
        })
        
        # Test that it handles zero division gracefully
        result = normalize_per_ha(df)
        
        # Check that result contains inf or nan for zero area
        assert np.isinf(result['fertilizer_per_ha'].iloc[0]) or np.isnan(result['fertilizer_per_ha'].iloc[0])
        assert np.isinf(result['pesticide_per_ha'].iloc[0]) or np.isnan(result['pesticide_per_ha'].iloc[0])
    
    def test_normalize_per_ha_missing_columns(self):
        """Test handling of missing required columns."""
        df = pd.DataFrame({
            'fertilizer': [100],
            'area': [2]
            # Missing 'pesticide' column
        })
        
        with pytest.raises(KeyError):
            normalize_per_ha(df)


class TestCreateFeatures:
    """Test cases for create_features function."""
    
    def test_create_features_basic(self):
        """Test basic feature creation."""
        df = pd.DataFrame({
            'fertilizer': [100, 200],
            'pesticide': [20, 40],
            'area': [2, 4],
            'rainfall': [1200, 1100],
            'temperature': [28.5, 29.1],
            'state': ['Odisha', 'Odisha'],
            'district': ['Cuttack', 'Puri'],
            'crop': ['Rice', 'Rice'],
            'season': ['Kharif', 'Kharif'],
            'soil_type': ['Clay', 'Sandy'],
            'irrigation': ['Flood', 'Sprinkler'],
            'seed_variety': ['HYV', 'HYV'],
            'yield': [1.68, 1.59]
        })
        
        X, y, numeric_features, categorical_features = create_features(df)
        
        # Check that X and y are created
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        
        # Check feature lists
        assert len(numeric_features) == 5  # area, rainfall, temperature, fertilizer_per_ha, pesticide_per_ha
        assert len(categorical_features) == 7  # state, district, crop, season, soil_type, irrigation, seed_variety
        
        # Check that per-hectare features are included
        assert 'fertilizer_per_ha' in X.columns
        assert 'pesticide_per_ha' in X.columns
        
        # Check target
        assert y.name == 'yield'
        assert len(y) == 2
    
    def test_create_features_missing_yield(self):
        """Test handling of missing yield column."""
        df = pd.DataFrame({
            'fertilizer': [100],
            'pesticide': [20],
            'area': [2],
            'rainfall': [1200],
            'temperature': [28.5],
            'state': ['Odisha'],
            'district': ['Cuttack'],
            'crop': ['Rice'],
            'season': ['Kharif'],
            'soil_type': ['Clay'],
            'irrigation': ['Flood'],
            'seed_variety': ['HYV']
            # Missing 'yield' column
        })
        
        with pytest.raises(KeyError):
            create_features(df)


class TestFillMissingValues:
    """Test cases for fill_missing_values function."""
    
    def test_fill_missing_values_numeric(self):
        """Test filling missing numeric values."""
        df = pd.DataFrame({
            'fertilizer': [100, np.nan, 150],
            'pesticide': [20, 40, np.nan],
            'area': [2, 4, 3]
        })
        
        result = fill_missing_values(df)
        
        # Check that NaN values are filled
        assert not result['fertilizer'].isnull().any()
        assert not result['pesticide'].isnull().any()
        
        # Check that median is used for numeric columns
        assert result['fertilizer'].iloc[1] == 125.0  # median of [100, 150]
        assert result['pesticide'].iloc[2] == 30.0    # median of [20, 40]
    
    def test_fill_missing_values_categorical(self):
        """Test filling missing categorical values."""
        df = pd.DataFrame({
            'crop': ['Rice', np.nan, 'Wheat', 'Rice'],
            'season': ['Kharif', 'Rabi', np.nan, 'Kharif']
        })
        
        result = fill_missing_values(df)
        
        # Check that NaN values are filled
        assert not result['crop'].isnull().any()
        assert not result['season'].isnull().any()
        
        # Check that mode is used for categorical columns
        assert result['crop'].iloc[1] == 'Rice'  # mode
        assert result['season'].iloc[2] == 'Kharif'  # mode


class TestValidateInputData:
    """Test cases for validate_input_data function."""
    
    def test_validate_input_data_valid(self):
        """Test validation of valid data."""
        df = pd.DataFrame({
            'area': [2, 4],
            'rainfall': [1200, 1100],
            'temperature': [28.5, 29.1],
            'fertilizer': [100, 200],
            'pesticide': [20, 40],
            'yield': [1.68, 1.59]
        })
        
        result = validate_input_data(df)
        
        assert result['is_valid'] == True
        assert len(result['errors']) == 0
    
    def test_validate_input_data_missing_columns(self):
        """Test validation with missing required columns."""
        df = pd.DataFrame({
            'area': [2, 4],
            'rainfall': [1200, 1100]
            # Missing required columns: temperature, fertilizer, pesticide, yield
        })
        
        result = validate_input_data(df)
        
        assert result['is_valid'] == False
        assert len(result['errors']) > 0
        assert 'Missing required columns' in result['errors'][0]
    
    def test_validate_input_data_negative_values(self):
        """Test validation with negative values."""
        df = pd.DataFrame({
            'area': [2, -1],  # Negative area
            'rainfall': [1200, 1100],
            'temperature': [28.5, 29.1],
            'fertilizer': [100, 200],
            'pesticide': [20, 40],
            'yield': [1.68, 1.59]
        })
        
        result = validate_input_data(df)
        
        assert result['is_valid'] == True  # Still valid, but with warnings
        assert len(result['warnings']) > 0
        assert 'negative values' in result['warnings'][0]


class TestGetFeatureSummary:
    """Test cases for get_feature_summary function."""
    
    def test_get_feature_summary_basic(self):
        """Test basic feature summary functionality."""
        df = pd.DataFrame({
            'area': [2, 4, 3],
            'rainfall': [1200, 1100, 1150],
            'crop': ['Rice', 'Wheat', 'Rice'],
            'season': ['Kharif', 'Rabi', 'Kharif']
        })
        
        result = get_feature_summary(df)
        
        # Check that summary contains expected keys
        assert 'shape' in result
        assert 'numeric_summary' in result
        assert 'categorical_summary' in result
        assert 'missing_values' in result
        
        # Check shape
        assert result['shape'] == (3, 4)
        
        # Check categorical summary
        assert 'crop' in result['categorical_summary']
        assert 'season' in result['categorical_summary']
        
        # Check that most common values are identified
        assert result['categorical_summary']['crop']['most_common'] == 'Rice'
        assert result['categorical_summary']['season']['most_common'] == 'Kharif'


if __name__ == "__main__":
    pytest.main([__file__])
