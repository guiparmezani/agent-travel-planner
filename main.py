from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph.message import add_messages # helper function to add messages to the state
from dotenv import load_dotenv
from langchain_core.tools import tool
from geopy.geocoders import Nominatim
from pydantic import BaseModel, Field
from datetime import datetime
from langchain_google_genai import ChatGoogleGenerativeAI

import os
import requests
import sys
import json

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

class AgentState(TypedDict):
    """The state of the agent."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    number_of_steps: int

geolocator = Nominatim(user_agent="weather-app")

class SearchInput(BaseModel):
    location:str = Field(description="The city and state, e.g., San Francisco")
    date:str = Field(description="the forecasting date for when to get the weather format (yyyy-mm-dd)")

@tool("get_weather_forecast", args_schema=SearchInput)
def get_weather_forecast(location: str, date: str):
    """Retrieves the weather using Open-Meteo API for a given location (city) and a date (yyyy-mm-dd). Returns a formatted weather forecast."""
    print(f"🌍 Looking up weather for {location} on {date}...")
    
    geocoded_location = geolocator.geocode(location)
    if geocoded_location:
        try:
            response = requests.get(f"https://api.open-meteo.com/v1/forecast?latitude={geocoded_location.latitude}&longitude={geocoded_location.longitude}&hourly=temperature_2m&start_date={date}&end_date={date}")
            data = response.json()
            
            if "hourly" in data and "time" in data["hourly"] and "temperature_2m" in data["hourly"]:
                weather_data = {}
                for time, temp in zip(data["hourly"]["time"], data["hourly"]["temperature_2m"]):
                    # Format time to be more readable
                    formatted_time = datetime.fromisoformat(time.replace('Z', '+00:00')).strftime('%H:%M')
                    weather_data[formatted_time] = f"{temp}°C"
                
                return {
                    "location": f"{location} ({geocoded_location.latitude:.2f}, {geocoded_location.longitude:.2f})",
                    "date": date,
                    "forecast": weather_data
                }
            else:
                return {"error": "No weather data available for the specified date"}
        except Exception as e:
            return {"error": f"Failed to fetch weather data: {str(e)}"}
    else:
        return {"error": f"Location '{location}' not found"}

tools = [get_weather_forecast]

def execute_tool_call(tool_call):
    """Execute a tool call and return the result."""
    tool_name = tool_call["name"]
    tool_args = tool_call["args"]
    
    if tool_name == "get_weather_forecast":
        return get_weather_forecast.invoke(tool_args)
    else:
        return {"error": f"Unknown tool: {tool_name}"}

def format_weather_output(weather_data):
    """Format weather data for nice display."""
    if "error" in weather_data:
        return f"❌ {weather_data['error']}"
    
    output = f"🌤️  Weather Forecast for {weather_data['location']}\n"
    output += f"📅 Date: {weather_data['date']}\n"
    output += "=" * 50 + "\n"
    
    for time, temp in weather_data['forecast'].items():
        output += f"🕐 {time}: {temp}\n"
    
    return output

def run_weather_agent(location, date=None):
    """Run the weather agent with the given location and date."""
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")
    
    # Create LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.1,  # Lower temperature for more consistent results
        max_retries=2,
        google_api_key=api_key,
    )
    
    # Bind tools to the model
    model = llm.bind_tools([get_weather_forecast])
    
    # Create the user message
    user_message = f"What is the weather in {location} on {date}?"
    
    print(f"🤖 Processing request: {user_message}")
    
    # Get the model response
    response = model.invoke([HumanMessage(content=user_message)])
    
    # Check if the model wants to call a tool
    if hasattr(response, 'tool_calls') and response.tool_calls:
        print("🔧 Executing tool call...")
        
        # Execute the tool call
        tool_result = execute_tool_call(response.tool_calls[0])
        
        # Format and display the result
        formatted_output = format_weather_output(tool_result)
        print("\n" + formatted_output)
        
        return tool_result
    else:
        # If no tool call, just return the response content
        print(f"🤖 {response.content}")
        return response.content

def main():
    """Main function to handle command line arguments."""
    if len(sys.argv) < 2:
        print("Usage: python main.py <location> [date]")
        print("Example: python main.py 'Joinville' '2025-01-15'")
        print("Date format: YYYY-MM-DD (optional, defaults to today)")
        sys.exit(1)
    
    location = sys.argv[1]
    date = sys.argv[2] if len(sys.argv) > 2 else None
    
    try:
        run_weather_agent(location, date)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()