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
    try:
        print("🤖 Trying Gemini LLM with tools...")
        
        signal.signal(signal.SIGALRM, timeout_handler)
        
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.7,
            google_api_key=gemini_api_key,
        )
        
        model = llm.bind_tools(tools)
        
        signal.alarm(0)
        print("✅ Gemini LLM with tools successful")
        return model
        
    except Exception as e:
        print(f"⚠️ Gemini failed: {e}")
        
        if not openai_api_key:
            raise Exception("Gemini timed out and no OpenAI API key provided")
        
        print("🤖 Falling back to OpenAI...")
        try:
            llm = ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0.7,
                openai_api_key=openai_api_key,
            )
            model = llm.bind_tools(tools)
            print("✅ OpenAI LLM with tools successful")
            return model
            
        except Exception as e2:
            raise Exception(f"Both Gemini and OpenAI failed. Gemini error: {e}, OpenAI error: {e2}")

class TimeoutError(Exception):
    """Custom timeout exception."""
    pass

def timeout_handler(signum, frame):
    """Signal handler for timeout."""
    raise TimeoutError("Operation timed out")

model = create_model()

tools_by_name = {tool.name: tool for tool in tools}

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
    response = model.invoke(state["messages"], config)
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
    graph = create_forecast_graph()
    
    if conversation_state:
        inputs = {
            "messages": conversation_state["messages"] + [("user", user_prompt)],
            "number_of_steps": conversation_state.get("number_of_steps", 0)
        }
    else:
        inputs = {"messages": [("user", user_prompt)], "number_of_steps": 0}
    
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