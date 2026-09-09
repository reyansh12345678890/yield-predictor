# AI-Powered Crop Yield Prediction and Optimization

A comprehensive machine learning system for predicting crop yield and providing actionable recommendations to farmers. This system uses historical agricultural data, weather information, and advanced ML algorithms to help farmers optimize their crop production.

## 🌾 Features

- **Yield Prediction**: Accurate crop yield prediction using Random Forest and XGBoost models
- **Optimization Recommendations**: AI-powered suggestions to improve yield by 10%+ through:
  - Fertilizer optimization
  - Irrigation method improvements
  - Crop selection optimization
- **Weather Integration**: Real-time weather data from OpenWeather API with mock fallback
- **Multilingual Support**: English and Odia language support for local farmers
- **Interactive UI**: User-friendly Streamlit interface with what-if analysis
- **REST API**: FastAPI-based API for integration with other systems
- **Batch Processing**: CSV upload for multiple predictions

## 📁 Project Structure

```
crop-yield/
├── data/
│   └── processed/
│       └── train_small.csv                # Sample training data
├── notebooks/
│   └── model_training.ipynb              # Complete training pipeline
├── src/
│   ├── ml/
│   │   ├── features.py                   # Feature engineering
│   │   ├── train.py                      # Training script
│   │   └── models/
│   │       ├── yield_model.pkl            # Trained model (generated)
│   │       └── feature_list.pkl           # Feature information (generated)
│   ├── api/
│   │   ├── weather_client.py             # Weather data integration
│   │   └── app.py                        # FastAPI application
│   └── inference/
│       └── predictor.py                  # Prediction and recommendations
├── frontend/
│   └── app.py                            # Streamlit UI
├── tests/
│   ├── test_featurizer.py                # Feature engineering tests
│   └── test_predict_smoke.py             # Prediction smoke tests
├── requirements.txt                       # Python dependencies
└── README.md                             # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd crop-yield

# Install dependencies
pip install -r requirements.txt
```

### 2. Train the Model

```bash
# Option 1: Run the training script
python src/ml/train.py data/processed/train_small.csv

# Option 2: Run the Jupyter notebook
jupyter notebook notebooks/model_training.ipynb
```

### 3. Start the API Server

```bash
uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### 4. Start the Frontend

```bash
streamlit run frontend/app.py
```

The UI will be available at `http://localhost:8501`

## 📊 Data Format

### Input Data Requirements

The system expects the following input parameters:

| Parameter | Type | Description | Units |
|-----------|------|-------------|-------|
| `district` | string | District name | - |
| `crop` | string | Crop type (Rice, Wheat, Corn, etc.) | - |
| `season` | string | Season (Kharif, Rabi, Annual) | - |
| `area` | float | Farm area | hectares |
| `irrigation` | string | Irrigation method (Flood, Sprinkler, Drip) | - |
| `fertilizer` | float | Fertilizer amount | kg |
| `pesticide` | float | Pesticide amount | kg |
| `year` | int | Year (optional, defaults to 2023) | - |
| `rainfall` | float | Rainfall (optional, fetched from API) | mm |
| `temperature` | float | Temperature (optional, fetched from API) | °C |
| `soil_type` | string | Soil type (optional, defaults to Clay) | - |
| `seed_variety` | string | Seed variety (optional, defaults to HYV) | - |

### Output Format

The system returns:

```json
{
  "predicted_yield": 1.68,
  "confidence_low": 1.45,
  "confidence_high": 1.91,
  "confidence_range": 0.46,
  "recommendations": [
    {
      "type": "fertilizer",
      "text": "Increase fertilizer application by 10% to 82.5 kg/ha",
      "expected_pct_gain": 8.5,
      "rationale": "Higher fertilizer application can boost yield by 8.5% while staying within safe limits"
    }
  ],
  "weather_data": {
    "district": "Cuttack",
    "year": 2023,
    "temperature": 28.5,
    "rainfall": 1200,
    "source": "mock_data"
  }
}
```

## 🔧 API Usage

### Single Prediction

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "district": "Cuttack",
    "crop": "Rice",
    "season": "Kharif",
    "area": 2.0,
    "irrigation": "Flood",
    "fertilizer": 150.0,
    "pesticide": 25.0,
    "year": 2023
  }'
```

### Batch Prediction

```bash
curl -X POST "http://localhost:8000/batch_predict" \
  -F "file=@your_data.csv"
```

### Weather Data

```bash
curl "http://localhost:8000/weather/Cuttack?year=2023"
```

## 🌤️ Weather Integration

The system integrates with OpenWeather API for real-time weather data:

1. **With API Key**: Set `OPENWEATHER_API_KEY` environment variable
2. **Without API Key**: Uses mock data for demonstration

### Mock Weather Data

The system includes realistic mock weather data for Odisha districts:

- **Cuttack**: 1200mm rainfall, 28.5°C average temperature
- **Puri**: 1100mm rainfall, 29.1°C average temperature  
- **Bhubaneswar**: 1300mm rainfall, 27.9°C average temperature

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_featurizer.py

# Run with verbose output
pytest tests/ -v
```

## 📈 Model Performance

The system trains two models and selects the best performing one:

- **Random Forest**: Ensemble method with good interpretability
- **XGBoost**: Gradient boosting with high accuracy

### Evaluation Metrics

- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Square Error  
- **R²**: Coefficient of determination

### Feature Importance

The system provides feature importance analysis using:
- Built-in feature importance for Random Forest
- SHAP values for XGBoost

## 🌍 Multilingual Support

The frontend supports both English and Odia languages:

- **English**: Full interface in English
- **Odia**: Localized interface for Odisha farmers

Switch languages using the sidebar language selector.

## 🔒 Safety Features

The recommendation engine includes safety constraints:

- **Fertilizer Limits**: Maximum safe fertilizer application per crop
- **Crop-Specific Rules**: Different limits for different crops
- **Conservative Recommendations**: Only suggests changes with >2% expected gain

## 🛠️ Development

### Adding New Crops

1. Add crop to `CROPS` list in `frontend/app.py`
2. Add safety limits in `src/inference/predictor.py`
3. Update crop descriptions in recommendation rationale

### Adding New Districts

1. Add district to `DISTRICTS` list in `frontend/app.py`
2. Add mock weather data in `src/api/weather_client.py`

### Extending Recommendations

Add new recommendation types in `src/inference/predictor.py`:

```python
def _optimize_new_feature(self, base_input, base_yield):
    # Implementation
    return recommendation_dict
```

## 📝 Example Usage

### 1. Train Model
```bash
python src/ml/train.py data/processed/train_small.csv
```

### 2. Start API
```bash
uvicorn src.api.app:app --reload
```

### 3. Start Frontend
```bash
streamlit run frontend/app.py
```

### 4. Make Predictions

1. Open `http://localhost:8501`
2. Select language (English/Odia)
3. Fill in farm details
4. Click "Predict Yield"
5. View predictions and recommendations
6. Use "What-If Analysis" to test scenarios

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- OpenWeather API for weather data
- Streamlit for the frontend framework
- FastAPI for the REST API
- Scikit-learn and XGBoost for machine learning
- The agricultural community for domain expertise

## 📞 Support

For questions or support, please open an issue in the repository or contact the development team.

---

**Built with ❤️ for the farming community**


