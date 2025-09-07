"""
Agent Travel Planner - Main Application
A modular travel planning application with specialized subagents.
"""

from dotenv import load_dotenv
from datetime import datetime
import sys

# Import subagents
from subagents.forecast import run_forecast_agent

# Load environment variables
load_dotenv()

def format_weather_output(result_data):
    """Format weather forecast data for display."""
    if not result_data.get("success", False):
        weather_data = result_data.get("weather_data", {})
        return f"❌ {weather_data.get('error', 'Unknown error occurred')}"
    
    weather_data = result_data.get("weather_data", {})
    location = result_data.get("location", "Unknown")
    date = result_data.get("date", "Unknown")
    
    output = f"🌤️  Weather Forecast for {weather_data.get('location', location)}\n"
    output += f"📅 Date: {date}\n"
    output += "=" * 50 + "\n"
    
    forecast = weather_data.get("forecast", {})
    for time, temp in forecast.items():
        output += f"🕐 {time}: {temp}\n"
    
    return output

def main():
    """Main function to handle command line arguments and route to appropriate subagents."""
    if len(sys.argv) < 2:
        print("🌍 Agent Travel Planner")
        print("=" * 50)
        print("Usage: python main.py <location> [date]")
        print("Example: python main.py 'Joinville' '2025-01-15'")
        print("Date format: YYYY-MM-DD (optional, defaults to today)")
        print("")
        print("Available features:")
        print("  • Weather forecast for any location")
        print("  • More travel planning features coming soon!")
        sys.exit(1)
    
    location = sys.argv[1]
    date = sys.argv[2] if len(sys.argv) > 2 else None
    
    try:
        print("🤖 Agent Travel Planner - Weather Forecast")
        print("=" * 50)
        
        # Route to weather forecast subagent
        result_data = run_forecast_agent(location, date)
        
        # Format and display the result
        formatted_output = format_weather_output(result_data)
        print("\n" + formatted_output)
        
        print("=" * 50)
        print("🎉 Travel planning session completed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()