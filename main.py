from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph.message import add_messages # helper function to add messages to the state
from langgraph.graph import StateGraph, END
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
    location: str
    date: str
    weather_data: dict
    formatted_output: str
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

# LangGraph Node Functions

def process_input(state: AgentState) -> AgentState:
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

def fetch_weather(state: AgentState) -> AgentState:
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

def format_output(state: AgentState) -> AgentState:
    """Format the weather data for display."""
    print("🎨 Formatting output...")
    
    weather_data = state["weather_data"]
    
    if "error" in weather_data:
        formatted_output = f"❌ {weather_data['error']}"
    else:
        output = f"🌤️  Weather Forecast for {weather_data['location']}\n"
        output += f"📅 Date: {weather_data['date']}\n"
        output += "=" * 50 + "\n"
        
        for time, temp in weather_data['forecast'].items():
            output += f"🕐 {time}: {temp}\n"
        
        formatted_output = output
    
    return {
        **state,
        "formatted_output": formatted_output,
        "number_of_steps": state.get("number_of_steps", 0) + 1
    }

def write_output(state: AgentState) -> AgentState:
    """Write the formatted output to console."""
    print("📤 Writing output...")
    print("\n" + state["formatted_output"])
    
    return {
        **state,
        "number_of_steps": state.get("number_of_steps", 0) + 1
    }

# Create the LangGraph workflow
def create_weather_graph():
    """Create the LangGraph workflow for weather forecasting."""
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("START", lambda state: state)  # Entry point
    workflow.add_node("process_input", process_input)
    workflow.add_node("fetch_weather", fetch_weather)
    workflow.add_node("format_output", format_output)
    workflow.add_node("write_output", write_output)
    workflow.add_node("END", lambda state: state)    # Exit point
    
    # Define the workflow edges
    workflow.set_entry_point("START")
    workflow.add_edge("START", "process_input")
    workflow.add_edge("process_input", "fetch_weather")
    workflow.add_edge("fetch_weather", "format_output")
    workflow.add_edge("format_output", "write_output")
    workflow.add_edge("write_output", "END")
    
    return workflow.compile()

def run_weather_agent(location, date=None):
    """Run the weather agent using LangGraph workflow."""
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")
    
    # Create the graph
    graph = create_weather_graph()
    
    # Create initial state
    initial_state = {
        "messages": [HumanMessage(content=f"What is the weather in {location} on {date}?")],
        "location": "",
        "date": "",
        "weather_data": {},
        "formatted_output": "",
        "number_of_steps": 0
    }
    
    print(f"🚀 Starting weather forecast workflow...")
    print(f"📍 Request: {location} on {date}")
    
    # Run the graph
    result = graph.invoke(initial_state)
    
    print("=" * 50)
    print(f"✅ Workflow completed in {result['number_of_steps']} steps")
    
    return result

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