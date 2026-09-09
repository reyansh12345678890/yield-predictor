"""
Streamlit frontend for AI-Powered Crop Yield Prediction.

This module provides a user-friendly interface for farmers to input their
farm data and get yield predictions with optimization recommendations.
Supports both English and Odia languages.
"""

import streamlit as st
import requests
import json
import pandas as pd
from typing import Dict, Any, List
import plotly.express as px
import plotly.graph_objects as go


# Language translations
TRANSLATIONS = {
    'en': {
        'title': 'AI-Powered Crop Yield Prediction',
        'subtitle': 'Get accurate yield predictions and optimization recommendations for your farm',
        'language': 'Language',
        'district': 'District',
        'crop': 'Crop',
        'season': 'Season',
        'area': 'Area (hectares)',
        'irrigation': 'Irrigation Method',
        'fertilizer': 'Fertilizer (kg)',
        'pesticide': 'Pesticide (kg)',
        'year': 'Year',
        'predict': 'Predict Yield',
        'predicted_yield': 'Predicted Yield',
        'confidence': 'Confidence Range',
        'recommendations': 'Optimization Recommendations',
        'what_if': 'What-If Analysis',
        'fertilizer_adjustment': 'Fertilizer Adjustment (%)',
        'update_prediction': 'Update Prediction',
        'yield_unit': 't/ha',
        'loading': 'Loading...',
        'error': 'Error',
        'success': 'Success',
        'no_recommendations': 'No specific recommendations available for this configuration.',
        'recommendation_text': 'Recommendation',
        'expected_gain': 'Expected Gain',
        'rationale': 'Rationale'
    },
    'odia': {
        'title': 'କୃଷି ଫସଲ ଉତ୍ପାଦନ ଭବିଷ୍ୟତବାଣୀ (AI ଚାଳିତ)',
        'subtitle': 'ଆପଣଙ୍କ କ୍ଷେତ୍ର ପାଇଁ ସଠିକ୍ ଫସଲ ଉତ୍ପାଦନ ଭବିଷ୍ୟତବାଣୀ ଏବଂ ଅପ୍ଟିମାଇଜେସନ୍ ସୁପାରିଶ ପାଆନ୍ତୁ',
        'language': 'ଭାଷା',
        'district': 'ଜିଲ୍ଲା',
        'crop': 'ଫସଲ',
        'season': 'ଋତୁ',
        'area': 'କ୍ଷେତ୍ର (ହେକ୍ଟର)',
        'irrigation': 'ଜଳସେଚନ ପଦ୍ଧତି',
        'fertilizer': 'ସାର (କିଲୋଗ୍ରାମ)',
        'pesticide': 'କୀଟନାଶକ (କିଲୋଗ୍ରାମ)',
        'year': 'ବର୍ଷ',
        'predict': 'ଫସଲ ଉତ୍ପାଦନ ଭବିଷ୍ୟତବାଣୀ',
        'predicted_yield': 'ଭବିଷ୍ୟତ ଫସଲ ଉତ୍ପାଦନ',
        'confidence': 'ଆତ୍ମବିଶ୍ୱାସ ପରିସର',
        'recommendations': 'ଅପ୍ଟିମାଇଜେସନ୍ ସୁପାରିଶ',
        'what_if': 'କଣ ହେବ ଯଦି ବିଶ୍ଳେଷଣ',
        'fertilizer_adjustment': 'ସାର ପରିବର୍ତ୍ତନ (%)',
        'update_prediction': 'ଭବିଷ୍ୟତବାଣୀ ଅପଡେଟ୍ କରନ୍ତୁ',
        'yield_unit': 'ଟନ୍/ହେକ୍ଟର',
        'loading': 'ଲୋଡ୍ ହେଉଛି...',
        'error': 'ତ୍ରୁଟି',
        'success': 'ସଫଳତା',
        'no_recommendations': 'ଏହି ବିନ୍ୟାସ ପାଇଁ କୌଣସି ନିର୍ଦ୍ଦିଷ୍ଟ ସୁପାରିଶ ଉପଲବ୍ଧ ନାହିଁ।',
        'recommendation_text': 'ସୁପାରିଶ',
        'expected_gain': 'ଆଶାକରାଯାଉଥିବା ଲାଭ',
        'rationale': 'ଯୁକ୍ତି'
    }
}

# Available options
DISTRICTS = [
    'Angul', 'Balangir', 'Balasore', 'Bargarh', 'Bhadrak', 'Bhubaneswar', 
    'Boudh', 'Cuttack', 'Deogarh', 'Dhenkanal', 'Gajapati', 'Ganjam', 
    'Jagatsinghpur', 'Jajpur', 'Jharsuguda', 'Kalahandi', 'Kandhamal', 
    'Kendrapara', 'Kendujhar', 'Khordha', 'Koraput', 'Malkangiri', 
    'Mayurbhanj', 'Nabarangpur', 'Nayagarh', 'Nuapada', 'Puri', 
    'Rayagada', 'Sambalpur', 'Subarnapur', 'Sundargarh'
]
CROPS = ['Rice', 'Wheat', 'Corn', 'Cotton', 'Sugarcane', 'Pulses', 'Oilseeds']
SEASONS = ['Kharif', 'Rabi', 'Annual']
IRRIGATION_METHODS = ['Flood', 'Sprinkler', 'Drip']
SOIL_TYPES = ['Clay', 'Sandy', 'Loam']
SEED_VARIETIES = ['HYV', 'Traditional']

# API configuration
API_BASE_URL = "http://localhost:8000"


def get_translation(key: str, language: str = 'en') -> str:
    """Get translation for a key in the specified language."""
    return TRANSLATIONS.get(language, TRANSLATIONS['en']).get(key, key)


def call_prediction_api(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Call the prediction API with input data."""
    try:
        response = requests.post(f"{API_BASE_URL}/predict", json=input_data, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {str(e)}")
        return None


def create_yield_chart(prediction_data: Dict[str, Any], language: str = 'en') -> go.Figure:
    """Create a yield prediction chart."""
    predicted_yield = prediction_data['predicted_yield']
    confidence_low = prediction_data['confidence_low']
    confidence_high = prediction_data['confidence_high']
    
    fig = go.Figure()
    
    # Add confidence interval
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[confidence_low, confidence_low],
        fill=None,
        mode='lines',
        line_color='rgba(0,100,80,0.2)',
        showlegend=False,
        hoverinfo='skip'
    ))
    
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[confidence_high, confidence_high],
        fill='tonexty',
        mode='lines',
        line_color='rgba(0,100,80,0.2)',
        name=get_translation('confidence', language),
        hoverinfo='skip'
    ))
    
    # Add predicted yield line
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[predicted_yield, predicted_yield],
        mode='lines',
        line=dict(color='red', width=3),
        name=f"{get_translation('predicted_yield', language)}: {predicted_yield:.2f} {get_translation('yield_unit', language)}"
    ))
    
    fig.update_layout(
        title=f"{get_translation('predicted_yield', language)} ({get_translation('yield_unit', language)})",
        xaxis=dict(showgrid=False, showticklabels=False),
        yaxis=dict(title=get_translation('yield_unit', language)),
        height=300,
        showlegend=True
    )
    
    return fig


def display_recommendations(recommendations: List[Dict[str, Any]], language: str = 'en'):
    """Display optimization recommendations."""
    if not recommendations:
        st.info(get_translation('no_recommendations', language))
        return
    
    st.subheader(get_translation('recommendations', language))
    
    for i, rec in enumerate(recommendations, 1):
        with st.expander(f"#{i} {rec.get('text', '')}"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    get_translation('expected_gain', language),
                    f"+{rec.get('expected_pct_gain', 0):.1f}%"
                )
            
            with col2:
                st.write(f"**{get_translation('rationale', language)}:**")
                st.write(rec.get('rationale', ''))


def main():
    """Main Streamlit application."""
    # Page configuration
    st.set_page_config(
        page_title="AI-Powered Crop Yield Prediction",
        page_icon="🌾",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    if 'language' not in st.session_state:
        st.session_state.language = 'en'
    if 'prediction_result' not in st.session_state:
        st.session_state.prediction_result = None
    
    # Sidebar for language selection
    with st.sidebar:
        st.title("🌾 " + get_translation('title', st.session_state.language))
        
        # Language selector
        language = st.selectbox(
            get_translation('language', st.session_state.language),
            ['en', 'odia'],
            index=0 if st.session_state.language == 'en' else 1,
            format_func=lambda x: 'English' if x == 'en' else 'ଓଡ଼ିଆ'
        )
        
        if language != st.session_state.language:
            st.session_state.language = language
            st.rerun()
        
        st.markdown("---")
        st.markdown(get_translation('subtitle', language))
    
    # Main content
    st.title("🌾 " + get_translation('title', language))
    st.markdown(get_translation('subtitle', language))
    
    # Create two columns for input form and results
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📝 " + get_translation('predict', language))
        
        # Input form
        with st.form("prediction_form"):
            district = st.selectbox(
                get_translation('district', language),
                DISTRICTS
            )
            
            crop = st.selectbox(
                get_translation('crop', language),
                CROPS
            )
            
            season = st.selectbox(
                get_translation('season', language),
                SEASONS
            )
            
            area = st.number_input(
                get_translation('area', language),
                min_value=0.1,
                max_value=100.0,
                value=2.0,
                step=0.1
            )
            
            irrigation = st.selectbox(
                get_translation('irrigation', language),
                IRRIGATION_METHODS
            )
            
            fertilizer = st.number_input(
                get_translation('fertilizer', language),
                min_value=0.0,
                max_value=1000.0,
                value=150.0,
                step=10.0
            )
            
            pesticide = st.number_input(
                get_translation('pesticide', language),
                min_value=0.0,
                max_value=100.0,
                value=25.0,
                step=5.0
            )
            
            year = st.number_input(
                get_translation('year', language),
                min_value=2020,
                max_value=2030,
                value=2023,
                step=1
            )
            
            submitted = st.form_submit_button(get_translation('predict', language))
            
            if submitted:
                # Prepare input data
                input_data = {
                    'district': district,
                    'crop': crop,
                    'season': season,
                    'area': area,
                    'irrigation': irrigation,
                    'fertilizer': fertilizer,
                    'pesticide': pesticide,
                    'year': year
                }
                
                # Show loading spinner
                with st.spinner(get_translation('loading', language)):
                    prediction_result = call_prediction_api(input_data)
                
                if prediction_result:
                    st.session_state.prediction_result = prediction_result
                    st.session_state.input_data = input_data
                    st.success(get_translation('success', language))
                else:
                    st.error(get_translation('error', language))
    
    with col2:
        if st.session_state.prediction_result:
            st.header("📊 " + get_translation('predicted_yield', language))
            
            # Display prediction results
            prediction_data = st.session_state.prediction_result
            
            # Create yield chart
            yield_chart = create_yield_chart(prediction_data, language)
            st.plotly_chart(yield_chart, use_container_width=True)
            
            # Display metrics
            col_metric1, col_metric2, col_metric3 = st.columns(3)
            
            with col_metric1:
                st.metric(
                    get_translation('predicted_yield', language),
                    f"{prediction_data['predicted_yield']:.2f} {get_translation('yield_unit', language)}"
                )
            
            with col_metric2:
                st.metric(
                    "Confidence Low",
                    f"{prediction_data['confidence_low']:.2f} {get_translation('yield_unit', language)}"
                )
            
            with col_metric3:
                st.metric(
                    "Confidence High",
                    f"{prediction_data['confidence_high']:.2f} {get_translation('yield_unit', language)}"
                )
            
            # Display weather data
            weather_data = prediction_data.get('weather_data', {})
            if weather_data:
                st.subheader("🌤️ Weather Data")
                col_w1, col_w2 = st.columns(2)
                
                with col_w1:
                    st.metric("Temperature", f"{weather_data.get('temperature', 0):.1f}°C")
                
                with col_w2:
                    st.metric("Rainfall", f"{weather_data.get('rainfall', 0):.0f} mm")
    
    # Recommendations section
    if st.session_state.prediction_result:
        st.markdown("---")
        recommendations = st.session_state.prediction_result.get('recommendations', [])
        display_recommendations(recommendations, language)
    
    # What-if analysis section
    if st.session_state.prediction_result:
        st.markdown("---")
        st.header("🔬 " + get_translation('what_if', language))
        
        col_whatif1, col_whatif2 = st.columns([1, 1])
        
        with col_whatif1:
            fertilizer_adjustment = st.slider(
                get_translation('fertilizer_adjustment', language),
                min_value=-20,
                max_value=20,
                value=0,
                step=5
            )
            
            if st.button(get_translation('update_prediction', language)):
                if st.session_state.input_data:
                    # Create modified input data
                    modified_input = st.session_state.input_data.copy()
                    modified_input['fertilizer'] = modified_input['fertilizer'] * (1 + fertilizer_adjustment / 100)
                    
                    # Get new prediction
                    with st.spinner(get_translation('loading', language)):
                        new_prediction = call_prediction_api(modified_input)
                    
                    if new_prediction:
                        st.session_state.whatif_result = new_prediction
        
        with col_whatif2:
            if 'whatif_result' in st.session_state:
                whatif_data = st.session_state.whatif_result
                
                st.subheader("Updated Prediction")
                
                # Calculate change
                original_yield = st.session_state.prediction_result['predicted_yield']
                new_yield = whatif_data['predicted_yield']
                change = new_yield - original_yield
                change_pct = (change / original_yield) * 100
                
                col_change1, col_change2 = st.columns(2)
                
                with col_change1:
                    st.metric(
                        "New Yield",
                        f"{new_yield:.2f} {get_translation('yield_unit', language)}"
                    )
                
                with col_change2:
                    st.metric(
                        "Change",
                        f"{change:+.2f} {get_translation('yield_unit', language)}",
                        f"{change_pct:+.1f}%"
                    )
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
            AI-Powered Crop Yield Prediction System | Built with Streamlit & FastAPI
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()


