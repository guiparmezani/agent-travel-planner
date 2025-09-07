"""
Weather Forecast Subagent
Handles all weather-related functionality using LangGraph workflow.
"""

from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from geopy.geocoders import Nominatim
from pydantic import BaseModel, Field
from datetime import datetime
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

import os
import requests
import signal
import time

load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

geolocator = Nominatim(user_agent="weather-app")

class AgentState(TypedDict):
    """The state of the agent."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    number_of_steps: int

class SearchInput(BaseModel):
    location: str = Field(description="The city and state, e.g., San Francisco")
    date: str = Field(description="the forecasting date for when to get the weather format (yyyy-mm-dd)")

@tool("get_weather_forecast", args_schema=SearchInput, return_direct=True)
def get_weather_forecast(location: str, date: str):
    """Retrieves the weather using Open-Meteo API for a given location (city) and a date (yyyy-mm-dd). Returns a list dictionary with the time and temperature for each hour."""
    location = geolocator.geocode(location)
    if location:
        try:
            response = requests.get(f"https://api.open-meteo.com/v1/forecast?latitude={location.latitude}&longitude={location.longitude}&hourly=temperature_2m&start_date={date}&end_date={date}")
            data = response.json()
            return {time: temp for time, temp in zip(data["hourly"]["time"], data["hourly"]["temperature_2m"])}
        except Exception as e:
            return {"error": str(e)}
    else:
        return {"error": "Location not found"}

tools = [get_weather_forecast]

def create_model():
    """Create LLM with tools using timeout and fallback system."""
    # Check if we have API keys first
    if not gemini_api_key and not openai_api_key:
        raise Exception("No API keys provided. Please set GEMINI_API_KEY or OPENAI_API_KEY in your .env file")
    
    # Try Gemini first if we have the key
    if gemini_api_key:
        try:
            print("🤖 Trying Gemini LLM with tools...")
            
            # Set up timeout
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(8)  # 8 second timeout
            
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                temperature=0.7,
                google_api_key=gemini_api_key,
            )
            
            model = llm.bind_tools(tools)
            
            # Test the API key with a simple call
            from langchain_core.messages import HumanMessage
            test_response = model.invoke([HumanMessage(content="test")])
            
            # Cancel timeout
            signal.alarm(0)
            print("✅ Gemini LLM with tools successful")
            return model
            
        except Exception as e:
            print(f"⚠️ Gemini failed: {e}")
            signal.alarm(0)  # Cancel timeout
            
            # Check if it's an authentication error
            error_msg = str(e).lower()
            if any(keyword in error_msg for keyword in ['invalid', 'unauthorized', 'authentication', 'api key', 'permission', 'forbidden']):
                print("🔑 Invalid Gemini API key detected, falling back to OpenAI...")
            else:
                print("🔄 Gemini error occurred, falling back to OpenAI...")
    else:
        print("⚠️ No Gemini API key provided, skipping Gemini...")
    
    # Fallback to OpenAI if we have the key
    if openai_api_key:
        print("🤖 Using OpenAI...")
        try:
            llm = ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0.7,
                openai_api_key=openai_api_key,
            )
            model = llm.bind_tools(tools)
            print("✅ OpenAI LLM with tools successful")
            return model
            
        except Exception as e:
            raise Exception(f"OpenAI failed: {e}")
    else:
        raise Exception("No OpenAI API key provided. Please set OPENAI_API_KEY in your .env file")

class TimeoutError(Exception):
    """Custom timeout exception."""
    pass

def timeout_handler(signum, frame):
    """Signal handler for timeout."""
    raise TimeoutError("Operation timed out")

# Global model variable - will be created when needed
model = None

tools_by_name = {tool.name: tool for tool in tools}

def get_model():
    """Get the model, creating it if it doesn't exist."""
    global model
    if model is None:
        model = create_model()
    return model

def call_tool(state: AgentState):
    outputs = []
    for tool_call in state["messages"][-1].tool_calls:
        tool_result = tools_by_name[tool_call["name"]].invoke(tool_call["args"])
        outputs.append(
            ToolMessage(
                content=tool_result,
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
            )
        )
    return {"messages": outputs}

def call_model(
    state: AgentState,
    config: RunnableConfig,
):
    current_model = get_model()
    response = current_model.invoke(state["messages"], config)
    return {"messages": [response]}

def should_continue(state: AgentState):
    messages = state["messages"]
    if not messages[-1].tool_calls:
        return "end"
    return "continue"

def create_forecast_graph():
    """Create the LangGraph workflow for weather forecasting."""
    workflow = StateGraph(AgentState)

    workflow.add_node("llm", call_model)
    workflow.add_node("tools", call_tool)
    
    workflow.set_entry_point("llm")
    
    workflow.add_conditional_edges(
        "llm",
        should_continue,
        {
            "continue": "tools",
            "end": END,
        },
    )
    
    workflow.add_edge("tools", "llm")

    return workflow.compile()

def run_forecast_agent(user_prompt, conversation_state=None):
    """Run the weather forecast agent using LangGraph workflow."""
    from langchain_core.messages import SystemMessage
    from datetime import datetime
    
    graph = create_forecast_graph()
    
    # Create system message for better date understanding
    current_date = datetime.now().strftime("%Y-%m-%d")
    current_year = datetime.now().year
    from datetime import timedelta
    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    
    system_prompt = f"""You are a helpful weather assistant. When users ask about weather, you should:

1. **Date Interpretation**: Be smart about understanding dates:
   - "today" = {current_date}
   - "tomorrow" = {tomorrow}
   - "September 8th" or "Sep 8" = {current_year}-09-08 (assume current year if no year specified)
   - "next week" = approximately 7 days from now
   - Any date without year = assume current year ({current_year})
   - If user says "2025-01-15", use that exact date

2. **Location Interpretation**: Be flexible with locations:
   - "Paris" = Paris, France
   - "New York" = New York, USA
   - "Tokyo" = Tokyo, Japan
   - Include country if ambiguous

3. **Response Format**: 
   - When giving temperatures for multiple days, format the response as a list of dictionaries with the date and temperature for each day.
   - When giving out a weather forecast for a single day, specify what day it is in a friendly format in the header of your response.

4. **Always use your get_weather_forecast tool** when users ask about weather. Don't ask for clarification unless absolutely necessary.

Current date context: {current_date} (Year: {current_year})"""
    
    if conversation_state:
        # Check if system message already exists
        has_system_message = any(isinstance(msg, SystemMessage) for msg in conversation_state["messages"])
        if has_system_message:
            # Add user message to existing conversation
            inputs = {
                "messages": conversation_state["messages"] + [("user", user_prompt)],
                "number_of_steps": conversation_state.get("number_of_steps", 0)
            }
        else:
            # Add system message and user message
            inputs = {
                "messages": [SystemMessage(content=system_prompt)] + conversation_state["messages"] + [("user", user_prompt)],
                "number_of_steps": conversation_state.get("number_of_steps", 0)
            }
    else:
        # First conversation - add system message
        inputs = {
            "messages": [SystemMessage(content=system_prompt), ("user", user_prompt)], 
            "number_of_steps": 0
        }
    
    print(f"🚀 Starting weather forecast workflow...")
    
    final_state = graph.invoke(inputs)
    
    print("✅ Forecast workflow completed")
    
    final_response = None
    for message in reversed(final_state["messages"]):
        if hasattr(message, 'content') and not (hasattr(message, 'tool_calls') and message.tool_calls):
            final_response = message.content
            break
    
    result_data = {
        "success": True,
        "llm_response": final_response,
        "steps_completed": final_state.get("number_of_steps", 0),
        "conversation_state": {
            "messages": final_state["messages"],
            "number_of_steps": final_state.get("number_of_steps", 0)
        }
    }
    
    return result_data