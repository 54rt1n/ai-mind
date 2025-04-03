# aim/tool/impl/weather.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0 

from typing import Dict, Any, List, Literal
from .base import ToolImplementation


class WeatherTool(ToolImplementation):
    """Implementation of the weather tool."""
    
    def weather_current(self, location: str, format: Literal["celsius", "fahrenheit"] = "celsius", **kwargs: Any) -> Dict[str, Any]:
        """Get current weather for a location.
        
        Args:
            location: City and state/country
            format: Temperature format (celsius/fahrenheit), defaults to "celsius"
            **kwargs: Additional parameters that will be ignored
            
        Returns:
            Dictionary with current weather information
        """
        # Mock implementation
        return {
            "temperature": 72 if format == "fahrenheit" else 22,
            "conditions": "Partly cloudy",
            "humidity": 65
        }
        
    def weather_forecast(self, location: str, days: int = 5, 
                        format: Literal["celsius", "fahrenheit"] = "celsius", **kwargs: Any) -> Dict[str, List[Dict[str, Any]]]:
        """Get weather forecast for a location.
        
        Args:
            location: City and state/country
            days: Number of days to forecast, defaults to 5
            format: Temperature format (celsius/fahrenheit), defaults to "celsius"
            **kwargs: Additional parameters that will be ignored
            
        Returns:
            Dictionary with forecast information
        """
        # Validate range constraint (not enforced by type system)
        if days < 1 or days > 10:
            raise ValueError("Days must be between 1 and 10")
        
        # Mock implementation for forecast
        forecast = []
        for i in range(days):
            temp = 70 + i if format == "fahrenheit" else 21 + (i / 2)
            forecast.append({
                "day": i + 1,
                "temperature": temp,
                "conditions": "Sunny" if i % 2 == 0 else "Cloudy"
            })
            
        return {"forecast": forecast}
    
    def execute(self, function_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the weather tool with given function and parameters.
        
        Args:
            function_name: Name of the function to execute
            parameters: Dictionary of parameter names and values
            
        Returns:
            Dictionary containing the weather response
            
        Raises:
            ValueError: If function is unknown or parameters are invalid
            RuntimeError: If weather service is unavailable
        """
        if function_name == "weather_current":
            return self.weather_current(
                location=parameters["location"],
                format=parameters.get("format", "celsius"),
                **parameters
            )
        elif function_name == "weather_forecast":
            return self.weather_forecast(
                location=parameters["location"],
                days=int(parameters.get("days", 5)),
                format=parameters.get("format", "celsius"),
                **parameters
            )
        else:
            raise ValueError(f"Unknown function: {function_name}") 