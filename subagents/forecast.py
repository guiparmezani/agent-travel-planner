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

# Load environment variables
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize geolocator
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

# Define tools list
tools = [get_weather_forecast]

# Create LLM with timeout and fallback system
def create_model():
    """Create LLM with tools using timeout and fallback system."""
    # Try Gemini first with timeout
    try:
        print("🤖 Trying Gemini LLM with tools...")
        
        # Set up timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(8)  # 8 second timeout
        
        # Create Gemini LLM
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.7,
            google_api_key=gemini_api_key,
        )
        
        # Bind tools to the model
        model = llm.bind_tools(tools)
        
        # Cancel timeout
        signal.alarm(0)
        print("✅ Gemini LLM with tools successful")
        return model
        
    except Exception as e:
        print(f"⚠️ Gemini failed: {e}")
        signal.alarm(0)  # Cancel timeout
        
        # Fallback to OpenAI
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

# Create the model with tools
model = create_model()

# Define tools by name for easy lookup
tools_by_name = {tool.name: tool for tool in tools}

# Define our tool node
def call_tool(state: AgentState):
    outputs = []
    # Iterate over the tool calls in the last message
    for tool_call in state["messages"][-1].tool_calls:
        # Get the tool by name
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
    # Invoke the model with the system prompt and the messages
    response = model.invoke(state["messages"], config)
    # We return a list, because this will get added to the existing messages state using the add_messages reducer
    return {"messages": [response]}

# Define the conditional edge that determines whether to continue or not
def should_continue(state: AgentState):
    messages = state["messages"]
    # If the last message is not a tool call, then we finish
    if not messages[-1].tool_calls:
        return "end"
    # default to continue
    return "continue"

# Create the LangGraph workflow
def create_forecast_graph():
    """Create the LangGraph workflow for weather forecasting."""
    # Define a new graph with our state
    workflow = StateGraph(AgentState)

    # 1. Add our nodes 
    workflow.add_node("llm", call_model)
    workflow.add_node("tools", call_tool)
    
    # 2. Set the entrypoint as `llm`, this is the first node called
    workflow.set_entry_point("llm")
    
    # 3. Add a conditional edge after the `llm` node is called.
    workflow.add_conditional_edges(
        # Edge is used after the `llm` node is called.
        "llm",
        # The function that will determine which node is called next.
        should_continue,
        # Mapping for where to go next, keys are strings from the function return, and the values are other nodes.
        # END is a special node marking that the graph is finish.
        {
            # If `continue`, then we call the tool node.
            "continue": "tools",
            # Otherwise we finish.
            "end": END,
        },
    )
    
    # 4. Add a normal edge after `tools` is called, `llm` node is called next.
    workflow.add_edge("tools", "llm")

    # Now we can compile our graph
    return workflow.compile()

def run_forecast_agent(user_prompt, conversation_state=None):
    """Run the weather forecast agent using LangGraph workflow."""
    # Create the graph
    graph = create_forecast_graph()
    
    # If we have conversation state, use it; otherwise create new state
    if conversation_state:
        # Add the new user message to existing conversation
        inputs = {
            "messages": conversation_state["messages"] + [("user", user_prompt)],
            "number_of_steps": conversation_state.get("number_of_steps", 0)
        }
    else:
        # Create our initial message dictionary with the user's prompt
        inputs = {"messages": [("user", user_prompt)], "number_of_steps": 0}
    
    print(f"🚀 Starting weather forecast workflow...")
    
    # Run the graph
    final_state = graph.invoke(inputs)
    
    print("✅ Forecast workflow completed")
    
    # Extract the final LLM response
    final_response = None
    for message in reversed(final_state["messages"]):
        if hasattr(message, 'content') and not (hasattr(message, 'tool_calls') and message.tool_calls):
            final_response = message.content
            break
    
    # Return structured result data with updated conversation state
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