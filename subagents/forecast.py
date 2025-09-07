"""
Weather Forecast Subagent
Handles all weather-related functionality using LangGraph workflow.
"""

from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langchain_core.tools import tool
from geopy.geocoders import Nominatim
from pydantic import BaseModel, Field
from datetime import datetime
from langchain_google_genai import ChatGoogleGenerativeAI

import os
import requests

# Load environment variables
api_key = os.getenv("GEMINI_API_KEY")

class ForecastState(TypedDict):
    """The state for the weather forecast subagent."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    location: str
    date: str
    weather_data: dict
    result_data: dict
    number_of_steps: int

# Initialize geocoder
geolocator = Nominatim(user_agent="weather-app")

class SearchInput(BaseModel):
    location: str = Field(description="The city and state, e.g., San Francisco")
    date: str = Field(description="the forecasting date for when to get the weather format (yyyy-mm-dd)")

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

# LangGraph Node Functions

def process_input(state: ForecastState) -> ForecastState:
    """Process the input and extract location and date information."""
    print("📝 Processing input...")
    
    # Get the user message
    user_message = state["messages"][-1].content
    
    # Create LLM to extract location and date
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.1,
        max_retries=2,
        google_api_key=api_key,
    )
    
    # Ask the LLM to extract location and date
    extraction_prompt = f"""
    Extract the location and date from this user request: "{user_message}"
    
    Return the information in this exact format:
    Location: [city name]
    Date: [YYYY-MM-DD format, or "today" if no date specified]
    
    If no date is specified, use "today".
    """
    
    response = llm.invoke(extraction_prompt)
    
    # Parse the response to extract location and date
    lines = response.content.strip().split('\n')
    location = None
    date = None
    
    for line in lines:
        if line.startswith('Location:'):
            location = line.replace('Location:', '').strip()
        elif line.startswith('Date:'):
            date_str = line.replace('Date:', '').strip()
            if date_str.lower() == 'today':
                date = datetime.now().strftime("%Y-%m-%d")
            else:
                date = date_str
    
    if not location:
        location = "Unknown"
    if not date:
        date = datetime.now().strftime("%Y-%m-%d")
    
    print(f"📍 Extracted - Location: {location}, Date: {date}")
    
    return {
        **state,
        "location": location,
        "date": date,
        "number_of_steps": state.get("number_of_steps", 0) + 1
    }

def fetch_weather(state: ForecastState) -> ForecastState:
    """Fetch weather data for the specified location and date."""
    print("🌍 Fetching weather data...")
    
    location = state["location"]
    date = state["date"]
    
    # Use the existing weather function
    weather_data = get_weather_forecast.invoke({"location": location, "date": date})
    
    print(f"✅ Weather data fetched for {location} on {date}")
    
    return {
        **state,
        "weather_data": weather_data,
        "number_of_steps": state.get("number_of_steps", 0) + 1
    }

def prepare_result(state: ForecastState) -> ForecastState:
    """Prepare the final result data for return to main application."""
    print("📤 Preparing result data...")
    
    weather_data = state["weather_data"]
    
    # Prepare structured result data
    result_data = {
        "success": "error" not in weather_data,
        "location": state["location"],
        "date": state["date"],
        "weather_data": weather_data,
        "steps_completed": state.get("number_of_steps", 0) + 1
    }
    
    return {
        **state,
        "result_data": result_data,
        "number_of_steps": state.get("number_of_steps", 0) + 1
    }

# Create the LangGraph workflow
def create_forecast_graph():
    """Create the LangGraph workflow for weather forecasting."""
    workflow = StateGraph(ForecastState)
    
    # Add nodes
    workflow.add_node("START", lambda state: state)  # Entry point
    workflow.add_node("process_input", process_input)
    workflow.add_node("fetch_weather", fetch_weather)
    workflow.add_node("prepare_result", prepare_result)
    workflow.add_node("END", lambda state: state)    # Exit point
    
    # Define the workflow edges
    workflow.set_entry_point("START")
    workflow.add_edge("START", "process_input")
    workflow.add_edge("process_input", "fetch_weather")
    workflow.add_edge("fetch_weather", "prepare_result")
    workflow.add_edge("prepare_result", "END")
    
    return workflow.compile()

def run_forecast_agent(location, date=None):
    """Run the weather forecast agent using LangGraph workflow."""
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")
    
    # Create the graph
    graph = create_forecast_graph()
    
    # Create initial state
    initial_state = {
        "messages": [HumanMessage(content=f"What is the weather in {location} on {date}?")],
        "location": "",
        "date": "",
        "weather_data": {},
        "result_data": {},
        "number_of_steps": 0
    }
    
    print(f"🚀 Starting weather forecast workflow...")
    print(f"📍 Request: {location} on {date}")
    
    # Run the graph
    result = graph.invoke(initial_state)
    
    print(f"✅ Forecast workflow completed in {result['number_of_steps']} steps")
    
    # Return the structured result data
    return result.get("result_data", {})
