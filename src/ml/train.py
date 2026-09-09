"""
Training script for crop yield prediction model.

This script can be run from the command line to train and save the model.
Usage: python src/ml/train.py data/processed/train_small.csv
"""

import pandas as pd
import numpy as np
import joblib
import argparse
import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

from ml.features import create_features, fill_missing_values, validate_input_data


def train_and_save(csv_path: str, output_model_path: str = None) -> dict:
    """
    Train crop yield prediction model and save it.
    
    Args:
        csv_path: Path to training CSV file
        output_model_path: Path to save the trained model (optional)
        
    Returns:
        Dictionary with training results
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    print(f"Loading data from: {csv_path}")
    
    # Load data
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded {df.shape[0]} samples with {df.shape[1]} features")
    except FileNotFoundError:
        raise FileNotFoundError(f"Training data not found at {csv_path}")
    
    # Validate input data
    validation_results = validate_input_data(df)
    if not validation_results['is_valid']:
        print("Data validation failed:")
        for error in validation_results['errors']:
            print(f"  - {error}")
        raise ValueError("Invalid input data")
    
    # Fill missing values
    df = fill_missing_values(df)
    
    # Create features
    X, y, numeric_features, categorical_features = create_features(df)
    
    print(f"Created feature matrix: {X.shape}")
    print(f"Numeric features: {numeric_features}")
    print(f"Categorical features: {categorical_features}")
    
    # Split data using time-based split
    train_mask = df['year'] <= 2022
    test_mask = df['year'] > 2022
    
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
        ]
    )
    
    # Create model pipelines
    rf_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    
    xgb_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0))
    ])
    
    # Train models
    print("Training Random Forest...")
    rf_pipeline.fit(X_train, y_train)
    
    print("Training XGBoost...")
    xgb_pipeline.fit(X_train, y_train)
    
    # Evaluate models
    def evaluate_model(model, X_test, y_test, model_name):
        y_pred = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        print(f"\n{model_name} Performance:")
        print(f"  MAE: {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  R²: {r2:.4f}")
        
        return {'mae': mae, 'rmse': rmse, 'r2': r2}
    
    rf_results = evaluate_model(rf_pipeline, X_test, y_test, "Random Forest")
    xgb_results = evaluate_model(xgb_pipeline, X_test, y_test, "XGBoost")
    
    # Select best model
    if rf_results['r2'] > xgb_results['r2']:
        best_model = rf_pipeline
        best_model_name = "Random Forest"
        best_results = rf_results
    else:
        best_model = xgb_pipeline
        best_model_name = "XGBoost"
        best_results = xgb_results
    
    print(f"\nBest model: {best_model_name} (R² = {best_results['r2']:.4f})")
    
    # Set default output path if not provided
    if output_model_path is None:
        output_model_path = "src/ml/models/yield_model.pkl"
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
    
    # Save the best model
    print(f"Saving model to: {output_model_path}")
    joblib.dump(best_model, output_model_path)
    
    # Save feature information
    feature_info = {
        'numeric_features': numeric_features,
        'categorical_features': categorical_features,
        'all_features': numeric_features + categorical_features,
        'target_column': 'yield'
    }
    
    feature_list_path = os.path.join(os.path.dirname(output_model_path), 'feature_list.pkl')
    print(f"Saving feature list to: {feature_list_path}")
    joblib.dump(feature_info, feature_list_path)
    
    # Return training results
    results = {
        'best_model_name': best_model_name,
        'best_model_path': output_model_path,
        'feature_list_path': feature_list_path,
        'performance': best_results,
        'training_samples': X_train.shape[0],
        'test_samples': X_test.shape[0],
        'features': {
            'numeric': numeric_features,
            'categorical': categorical_features,
            'total': len(numeric_features) + len(categorical_features)
        }
    }
    
    print("\nTraining completed successfully!")
    return results


def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(description='Train crop yield prediction model')
    parser.add_argument('csv_path', help='Path to training CSV file')
    parser.add_argument('--output', '-o', help='Output path for trained model', 
                       default='src/ml/models/yield_model.pkl')
    
    args = parser.parse_args()
    
    try:
        results = train_and_save(args.csv_path, args.output)
        
        print("\n" + "="*50)
        print("TRAINING SUMMARY")
        print("="*50)
        print(f"Best Model: {results['best_model_name']}")
        print(f"Model Path: {results['best_model_path']}")
        print(f"Feature List: {results['feature_list_path']}")
        print(f"Test R² Score: {results['performance']['r2']:.4f}")
        print(f"Test MAE: {results['performance']['mae']:.4f}")
        print(f"Test RMSE: {results['performance']['rmse']:.4f}")
        print(f"Total Features: {results['features']['total']}")
        print("="*50)
        
    except Exception as e:
        print(f"Training failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()


