"""
Weather client for fetching weather data from OpenWeather API.

This module provides functions to fetch weather data for agricultural districts,
with fallback to mock data when API is not available.
"""

import os
import requests
import json
from typing import Dict, Any, Optional
import random


class WeatherClient:
    """Client for fetching weather data from OpenWeather API."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize weather client.
        
        Args:
            api_key: OpenWeather API key (if None, will use mock data)
        """
        self.api_key = api_key or os.getenv('OPENWEATHER_API_KEY')
        self.base_url = "http://api.openweathermap.org/data/2.5"
        self.mock_data = self._get_mock_weather_data()
        
    def get_recent_weather(self, district: str, year: int = 2023) -> Dict[str, Any]:
        """
        Get recent weather data for a district.
        
        Args:
            district: Name of the district
            year: Year for which to get weather data
            
        Returns:
            Dictionary with weather data (rainfall, temperature)
        """
        if self.api_key:
            try:
                return self._fetch_from_api(district, year)
            except Exception as e:
                print(f"API fetch failed: {e}. Using mock data.")
                return self._get_mock_weather(district, year)
        else:
            print("No API key provided. Using mock weather data.")
            return self._get_mock_weather(district, year)
    
    def get_yearly_rainfall(self, district: str, year: int = 2023) -> Dict[str, Any]:
        """
        Get yearly rainfall data for a district.
        
        Args:
            district: Name of the district
            year: Year for which to get rainfall data
            
        Returns:
            Dictionary with rainfall data
        """
        weather_data = self.get_recent_weather(district, year)
        return {
            'district': district,
            'year': year,
            'rainfall': weather_data.get('rainfall', 0),
            'source': weather_data.get('source', 'mock')
        }
    
    def _fetch_from_api(self, district: str, year: int) -> Dict[str, Any]:
        """
        Fetch weather data from OpenWeather API.
        
        Args:
            district: Name of the district
            year: Year for which to get weather data
            
        Returns:
            Dictionary with weather data
        """
        # Note: OpenWeather API provides current weather, not historical data
        # For historical data, you would need a different API or service
        # This is a simplified example showing how to integrate with weather APIs
        
        # Get current weather for the district
        url = f"{self.base_url}/weather"
        params = {
            'q': f"{district},IN",  # Assuming India
            'appid': self.api_key,
            'units': 'metric'
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        # Extract relevant information
        temperature = data['main']['temp']
        
        # For rainfall, we'll use a mock value since current weather API
        # doesn't provide historical rainfall data
        rainfall = self._estimate_rainfall_from_api_data(data)
        
        return {
            'district': district,
            'year': year,
            'temperature': temperature,
            'rainfall': rainfall,
            'humidity': data['main']['humidity'],
            'source': 'openweather_api',
            'api_response': data
        }
    
    def _estimate_rainfall_from_api_data(self, api_data: Dict[str, Any]) -> float:
        """
        Estimate rainfall from API data (simplified approach).
        
        Args:
            api_data: Raw API response data
            
        Returns:
            Estimated rainfall in mm
        """
        # This is a simplified estimation - in practice, you'd need
        # historical weather data or a different API service
        humidity = api_data['main']['humidity']
        pressure = api_data['main']['pressure']
        
        # Simple heuristic based on humidity and pressure
        base_rainfall = 1000  # Base rainfall in mm/year
        humidity_factor = humidity / 100.0
        pressure_factor = (pressure - 1000) / 50.0
        
        estimated_rainfall = base_rainfall * humidity_factor * (1 + pressure_factor * 0.1)
        
        return max(500, min(2000, estimated_rainfall))  # Clamp between 500-2000mm
    
    def _get_mock_weather(self, district: str, year: int) -> Dict[str, Any]:
        """
        Get mock weather data for a district.
        
        Args:
            district: Name of the district
            year: Year for which to get weather data
            
        Returns:
            Dictionary with mock weather data
        """
        # Use deterministic random seed based on district and year
        random.seed(hash(f"{district}_{year}") % 2**32)
        
        # Get base data for the district
        base_data = self.mock_data.get(district, self.mock_data['default'])
        
        # Add some variation based on year
        year_factor = 1 + (year - 2020) * 0.02  # Slight trend over years
        
        rainfall = base_data['rainfall'] * year_factor * random.uniform(0.8, 1.2)
        temperature = base_data['temperature'] * random.uniform(0.95, 1.05)
        
        return {
            'district': district,
            'year': year,
            'temperature': round(temperature, 1),
            'rainfall': round(rainfall, 0),
            'humidity': random.randint(60, 90),
            'source': 'mock_data',
            'note': 'This is mock data for demonstration purposes'
        }
    
    def _get_mock_weather_data(self) -> Dict[str, Dict[str, float]]:
        """
        Get mock weather data for different districts.
        
        Returns:
            Dictionary with mock weather data for each district
        """
        return {
            'Cuttack': {
                'rainfall': 1200,
                'temperature': 28.5
            },
            'Puri': {
                'rainfall': 1100,
                'temperature': 29.1
            },
            'Bhubaneswar': {
                'rainfall': 1300,
                'temperature': 27.9
            },
            'default': {
                'rainfall': 1000,
                'temperature': 28.0
            }
        }


def get_recent_weather(district: str, year: int = 2023) -> Dict[str, Any]:
    """
    Convenience function to get recent weather data.
    
    Args:
        district: Name of the district
        year: Year for which to get weather data
        
    Returns:
        Dictionary with weather data
    """
    client = WeatherClient()
    return client.get_recent_weather(district, year)


def get_yearly_rainfall(district: str, year: int = 2023) -> Dict[str, Any]:
    """
    Convenience function to get yearly rainfall data.
    
    Args:
        district: Name of the district
        year: Year for which to get rainfall data
        
    Returns:
        Dictionary with rainfall data
    """
    client = WeatherClient()
    return client.get_yearly_rainfall(district, year)


def test_weather_client():
    """Test function to verify weather client functionality."""
    print("Testing Weather Client...")
    
    # Test with mock data
    client = WeatherClient()
    
    districts = ['Cuttack', 'Puri', 'Bhubaneswar']
    
    for district in districts:
        weather = client.get_recent_weather(district, 2023)
        print(f"\n{district} Weather (2023):")
        print(f"  Temperature: {weather['temperature']}°C")
        print(f"  Rainfall: {weather['rainfall']} mm")
        print(f"  Source: {weather['source']}")
    
    print("\nWeather client test completed!")


if __name__ == "__main__":
    test_weather_client()


