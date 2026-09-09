"""
Feature engineering functions for crop yield prediction.

This module contains functions for creating and transforming features
for the crop yield prediction model.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any


def normalize_per_ha(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize fertilizer and pesticide amounts by area.
    
    Args:
        df: Input DataFrame with fertilizer, pesticide and area columns
        
    Returns:
        DataFrame with additional per hectare columns
    """
    df = df.copy()
    
    if 'fertilizer' in df.columns and 'area' in df.columns:
        df['fertilizer_per_ha'] = df['fertilizer'] / df['area']
    
    if 'pesticide' in df.columns and 'area' in df.columns:
        df['pesticide_per_ha'] = df['pesticide'] / df['area']
        
    return df


def validate_input_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate input data meets requirements.
    
    Args:
        df: Input DataFrame to validate
        
    Returns:
        Dictionary with validation results
    """
    results = {
        'is_valid': True,
        'errors': [],
        'warnings': []
    }
    
    # Check required columns
    required_cols = ['area', 'rainfall', 'temperature', 'fertilizer', 'pesticide']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        results['is_valid'] = False
        results['errors'].append(f'Missing required columns: {", ".join(missing_cols)}')
    
    # Check for negative values
    numeric_cols = ['area', 'rainfall', 'temperature', 'fertilizer', 'pesticide']
    for col in numeric_cols:
        if col in df.columns:
            if (df[col] < 0).any():
                results['warnings'].append(f'Found negative values in {col}')
                
    return results


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values in the dataset.
    
    Args:
        df: Input DataFrame with missing values
        
    Returns:
        DataFrame with missing values filled
    """
    df = df.copy()
    
    # Fill numeric columns with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
    
    # Fill categorical columns with mode
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
    
    return df


def create_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    """
    Create feature matrix from input data.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Tuple containing:
        - Feature matrix X
        - Target vector y
        - List of numeric feature names
        - List of categorical feature names
    """
    if 'yield' not in df.columns:
        raise KeyError("DataFrame must contain 'yield' column")
    
    # Normalize per hectare values
    df = normalize_per_ha(df)
    
    # Define feature columns
    numeric_features = ['area', 'rainfall', 'temperature', 'fertilizer_per_ha', 'pesticide_per_ha']
    categorical_features = ['state', 'district', 'crop', 'season', 'soil_type', 'irrigation', 'seed_variety']
    
    # Create feature matrix X and target y
    X = df[numeric_features + categorical_features]
    y = df['yield']
    
    return X, y, numeric_features, categorical_features


def get_feature_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get summary statistics for features.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with feature summary statistics
    """
    summary = {
        'shape': df.shape,
        'numeric_summary': df.describe(),
        'categorical_summary': {
            col: df[col].value_counts()
            for col in df.select_dtypes(include=['object']).columns
        },
        'missing_values': df.isnull().sum()
    }
    
    return summary