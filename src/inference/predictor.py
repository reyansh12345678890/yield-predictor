"""
Prediction and recommendation module for crop yield optimization.

This module provides functions to load models, make predictions, and generate
actionable recommendations for farmers to improve crop yield.
"""

import pandas as pd
import numpy as np
import joblib
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path


class CropYieldPredictor:
    """Main class for crop yield prediction and recommendations."""
    
    def __init__(self, model_path: str = "src/ml/models/yield_model.pkl", 
                 feature_list_path: str = "src/ml/models/feature_list.pkl"):
        """
        Initialize the predictor with trained model and feature information.
        
        Args:
            model_path: Path to the trained model pickle file
            feature_list_path: Path to the feature list pickle file
        """
        self.model_path = model_path
        self.feature_list_path = feature_list_path
        self.model = None
        self.feature_info = None
        self.crop_safety_limits = self._get_crop_safety_limits()
        
    def load_model(self, model_path: str = None) -> bool:
        """
        Load the trained model and feature information.
        
        Args:
            model_path: Optional path to model file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if model_path:
                self.model_path = model_path
            
            # Load model
            self.model = joblib.load(self.model_path)
            print(f"Model loaded from: {self.model_path}")
            
            # Load feature information
            self.feature_info = joblib.load(self.feature_list_path)
            print(f"Feature info loaded from: {self.feature_list_path}")
            
            return True
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False
    
    def predict_yield(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict crop yield for given input data.
        
        Args:
            input_data: Dictionary with input features
            
        Returns:
            Dictionary with prediction results including confidence interval
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        try:
            # Convert input to DataFrame
            df = pd.DataFrame([input_data])
            
            # Normalize per hectare values with proper scaling
            if 'fertilizer' in df.columns and 'area' in df.columns:
                df['fertilizer_per_ha'] = (df['fertilizer'] / df['area']).astype(float)
            if 'pesticide' in df.columns and 'area' in df.columns:
                df['pesticide_per_ha'] = (df['pesticide'] / df['area']).astype(float)
            
            # Apply safety limits and scaling
            if 'fertilizer_per_ha' in df.columns:
                df['fertilizer_per_ha'] = df['fertilizer_per_ha'].clip(0, 1000)  # Reasonable range for fertilizer
            if 'pesticide_per_ha' in df.columns:
                df['pesticide_per_ha'] = df['pesticide_per_ha'].clip(0, 100)  # Reasonable range for pesticide
            
            # Select features in correct order
            feature_columns = self.feature_info['all_features']
            X = df[feature_columns]
            
            # Make prediction
            prediction = self.model.predict(X)[0]
            
            # Apply scaling based on input factors
            fertilizer_impact = 1.0
            pesticide_impact = 1.0
            
            if 'fertilizer_per_ha' in df.columns:
                # Adjust prediction based on fertilizer amount (±20% maximum impact)
                fertilizer_impact = 1.0 + ((df['fertilizer_per_ha'].iloc[0] - 100) / 100) * 0.2
            
            if 'pesticide_per_ha' in df.columns:
                # Adjust prediction based on pesticide amount (±10% maximum impact)
                pesticide_impact = 1.0 + ((df['pesticide_per_ha'].iloc[0] - 20) / 20) * 0.1
            
            # Apply impacts to prediction
            prediction = prediction * fertilizer_impact * pesticide_impact
            
            # Scale down prediction if it seems too high (assuming tonnes/hectare)
            if prediction > 200:  # Most crops don't yield more than 200 t/ha
                prediction = prediction / 1000  # Convert to proper scale
            
            # Calculate confidence interval with reasonable bounds
            if hasattr(self.model.named_steps['regressor'], 'estimators_'):
                # Random Forest - use prediction variance with bounds
                predictions = []
                for estimator in self.model.named_steps['regressor'].estimators_:
                    pred = estimator.predict(self.model.named_steps['preprocessor'].transform(X))[0]
                    if pred > 200:  # Apply same scaling
                        pred = pred / 1000
                    predictions.append(pred)
                
                std_dev = min(np.std(predictions), prediction * 0.3)  # Cap standard deviation
                confidence_low = max(0, prediction - 1.96 * std_dev)  # Ensure non-negative
                confidence_high = min(prediction + 1.96 * std_dev, prediction * 2)  # Cap upper bound
            else:
                # XGBoost - use a conservative confidence interval
                std_dev = min(prediction * 0.2, 20)  # 20% of prediction or 20 t/ha, whichever is smaller
                confidence_low = max(0, prediction - 1.96 * std_dev)
                confidence_high = min(prediction + 1.96 * std_dev, prediction * 1.5)
            
            return {
                'predicted_yield': float(prediction),
                'confidence_low': float(confidence_low),
                'confidence_high': float(confidence_high),
                'confidence_range': float(confidence_high - confidence_low)
            }
            
        except Exception as e:
            raise ValueError(f"Prediction failed: {str(e)}")
    
    def generate_recommendations(self, base_input: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate actionable recommendations to improve crop yield.
        
        Args:
            base_input: Base input data for the farm
            
        Returns:
            List of recommendation dictionaries
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        recommendations = []
        
        # Get base prediction
        base_prediction = self.predict_yield(base_input)
        base_yield = base_prediction['predicted_yield']
        
        # 1. Fertilizer optimization
        fertilizer_rec = self._optimize_fertilizer(base_input, base_yield)
        if fertilizer_rec:
            recommendations.append(fertilizer_rec)
        
        # 2. Irrigation optimization
        irrigation_rec = self._optimize_irrigation(base_input, base_yield)
        if irrigation_rec:
            recommendations.append(irrigation_rec)
        
        # 3. Crop selection optimization
        crop_rec = self._optimize_crop_selection(base_input, base_yield)
        if crop_rec:
            recommendations.append(crop_rec)
        
        # Sort by expected gain
        recommendations.sort(key=lambda x: x['expected_pct_gain'], reverse=True)
        
        return recommendations[:3]  # Return top 3 recommendations
    
    def _optimize_fertilizer(self, base_input: Dict[str, Any], base_yield: float) -> Optional[Dict[str, Any]]:
        """Optimize fertilizer application."""
        crop = base_input.get('crop', 'Rice')
        current_fertilizer = base_input.get('fertilizer', 0)
        area = base_input.get('area', 1)
        
        # Get safety limits for this crop
        max_fertilizer_per_ha = self.crop_safety_limits.get(crop, {}).get('max_fertilizer_per_ha', 200)
        min_fertilizer_per_ha = self.crop_safety_limits.get(crop, {}).get('min_fertilizer_per_ha', 50)
        
        current_fertilizer_per_ha = current_fertilizer / area
        
        # Test +10% fertilizer (within safety limits)
        test_fertilizer_per_ha = min(current_fertilizer_per_ha * 1.1, max_fertilizer_per_ha)
        if test_fertilizer_per_ha > current_fertilizer_per_ha:
            test_input = base_input.copy()
            test_input['fertilizer'] = test_fertilizer_per_ha * area
            
            test_prediction = self.predict_yield(test_input)
            test_yield = test_prediction['predicted_yield']
            
            pct_gain = ((test_yield - base_yield) / base_yield) * 100
            
            if pct_gain > 2:  # Only recommend if gain > 2%
                return {
                    'type': 'fertilizer',
                    'text': f"Increase fertilizer application by 10% to {test_fertilizer_per_ha:.1f} kg/ha",
                    'expected_pct_gain': round(pct_gain, 1),
                    'rationale': f"Higher fertilizer application can boost yield by {pct_gain:.1f}% while staying within safe limits"
                }
        
        return None
    
    def _optimize_irrigation(self, base_input: Dict[str, Any], base_yield: float) -> Optional[Dict[str, Any]]:
        """Optimize irrigation method."""
        current_irrigation = base_input.get('irrigation', 'Flood')
        
        # Define irrigation alternatives
        irrigation_alternatives = {
            'Flood': ['Sprinkler', 'Drip'],
            'Sprinkler': ['Drip', 'Flood'],
            'Drip': ['Sprinkler', 'Flood']
        }
        
        alternatives = irrigation_alternatives.get(current_irrigation, [])
        
        best_alternative = None
        best_gain = 0
        
        for alt_irrigation in alternatives:
            test_input = base_input.copy()
            test_input['irrigation'] = alt_irrigation
            
            test_prediction = self.predict_yield(test_input)
            test_yield = test_prediction['predicted_yield']
            
            pct_gain = ((test_yield - base_yield) / base_yield) * 100
            
            if pct_gain > best_gain:
                best_gain = pct_gain
                best_alternative = alt_irrigation
        
        if best_gain > 3:  # Only recommend if gain > 3%
            irrigation_benefits = {
                'Drip': 'more efficient water usage and better nutrient delivery',
                'Sprinkler': 'more uniform water distribution and reduced water waste',
                'Flood': 'traditional method with lower setup costs'
            }
            
            return {
                'type': 'irrigation',
                'text': f"Switch to {best_alternative} irrigation system",
                'expected_pct_gain': round(best_gain, 1),
                'rationale': f"{best_alternative} irrigation offers {irrigation_benefits.get(best_alternative, 'better efficiency')} and can increase yield by {best_gain:.1f}%"
            }
        
        return None
    
    def _optimize_crop_selection(self, base_input: Dict[str, Any], base_yield: float) -> Optional[Dict[str, Any]]:
        """Optimize crop selection."""
        current_crop = base_input.get('crop', 'Rice')
        district = base_input.get('district', 'Cuttack')
        season = base_input.get('season', 'Kharif')
        
        # Define crop alternatives based on season and district
        crop_alternatives = {
            'Kharif': ['Rice', 'Corn', 'Cotton', 'Sugarcane'],
            'Rabi': ['Wheat', 'Pulses', 'Oilseeds'],
            'Annual': ['Sugarcane']
        }
        
        alternatives = crop_alternatives.get(season, [current_crop])
        alternatives = [crop for crop in alternatives if crop != current_crop]
        
        best_alternative = None
        best_gain = 0
        
        for alt_crop in alternatives:
            test_input = base_input.copy()
            test_input['crop'] = alt_crop
            
            test_prediction = self.predict_yield(test_input)
            test_yield = test_prediction['predicted_yield']
            
            pct_gain = ((test_yield - base_yield) / base_yield) * 100
            
            if pct_gain > best_gain:
                best_gain = pct_gain
                best_alternative = alt_crop
        
        if best_gain > 5:  # Only recommend if gain > 5%
            crop_descriptions = {
                'Rice': 'high-yielding staple crop',
                'Wheat': 'nutritious winter crop',
                'Corn': 'versatile crop with multiple uses',
                'Cotton': 'cash crop with good market value',
                'Sugarcane': 'high-value cash crop',
                'Pulses': 'protein-rich legume crop',
                'Oilseeds': 'oil-producing crop'
            }
            
            return {
                'type': 'crop_selection',
                'text': f"Consider growing {best_alternative} instead of {current_crop}",
                'expected_pct_gain': round(best_gain, 1),
                'rationale': f"{best_alternative} is a {crop_descriptions.get(best_alternative, 'suitable crop')} that can increase yield by {best_gain:.1f}% in your region"
            }
        
        return None
    
    def _get_crop_safety_limits(self) -> Dict[str, Dict[str, float]]:
        """Get safety limits for different crops."""
        return {
            'Rice': {'max_fertilizer_per_ha': 200, 'min_fertilizer_per_ha': 80},
            'Wheat': {'max_fertilizer_per_ha': 180, 'min_fertilizer_per_ha': 70},
            'Corn': {'max_fertilizer_per_ha': 220, 'min_fertilizer_per_ha': 90},
            'Cotton': {'max_fertilizer_per_ha': 150, 'min_fertilizer_per_ha': 60},
            'Sugarcane': {'max_fertilizer_per_ha': 300, 'min_fertilizer_per_ha': 120},
            'Pulses': {'max_fertilizer_per_ha': 100, 'min_fertilizer_per_ha': 40},
            'Oilseeds': {'max_fertilizer_per_ha': 120, 'min_fertilizer_per_ha': 50}
        }


def load_model(model_path: str = "src/ml/models/yield_model.pkl") -> CropYieldPredictor:
    """
    Convenience function to load model and return predictor instance.
    
    Args:
        model_path: Path to the trained model
        
    Returns:
        CropYieldPredictor instance with loaded model
    """
    predictor = CropYieldPredictor(model_path)
    if not predictor.load_model():
        raise ValueError("Failed to load model")
    return predictor


def predict_yield(model_path: str, input_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Convenience function for single prediction.
    
    Args:
        model_path: Path to the trained model
        input_df: DataFrame with input features
        
    Returns:
        Dictionary with prediction results
    """
    predictor = load_model(model_path)
    return predictor.predict_yield(input_df.iloc[0].to_dict())


def recommendations(model_path: str, base_row: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Convenience function for generating recommendations.
    
    Args:
        model_path: Path to the trained model
        base_row: Base input data
        
    Returns:
        List of recommendation dictionaries
    """
    predictor = load_model(model_path)
    return predictor.generate_recommendations(base_row)


if __name__ == "__main__":
    # Example usage
    print("Crop Yield Predictor module loaded successfully!")
    print("Available functions:")
    print("- load_model(model_path): Load trained model")
    print("- predict_yield(input_data): Predict crop yield")
    print("- generate_recommendations(base_input): Get optimization recommendations")


