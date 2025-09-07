"""
Agent Travel Planner - Main Application
A modular travel planning application with specialized subagents.
"""

from datetime import datetime
import sys

# Import subagents
from subagents.forecast import run_forecast_agent

def format_weather_output(result_data):
    """Format LLM response for display."""
    if not result_data.get("success", False):
        return f"❌ Error occurred during processing"
    
    llm_response = result_data.get("llm_response", "No response received")
    
    # The LLM response should already be nicely formatted
    return llm_response

def main():
    """Main function to handle chat-style interaction with the travel planner."""
    print("🌍 Agent Travel Planner - Chat Mode")
    print("=" * 50)
    print("Welcome! I can help you with weather forecasts and travel planning.")
    print("Type your questions naturally, like:")
    print("  • 'What's the weather like in Paris today?'")
    print("  • 'How is the weather in Tokyo on 2025-01-15?'")
    print("  • 'Weather forecast for New York tomorrow'")
    print("")
    print("Type 'quit', 'exit', or 'bye' to end the session.")
    print("=" * 50)
    
    # Initialize conversation state
    conversation_state = None
    
    while True:
        try:
            # Get user input
            user_input = input("\n🤖 You: ").strip()
            
            # Check for exit commands
            if user_input.lower() in ['quit', 'exit', 'bye', 'q']:
                print("\n👋 Thanks for using Agent Travel Planner! Goodbye!")
                break
            
            # Skip empty inputs
            if not user_input:
                continue
            
            print(f"\n🔄 Processing: {user_input}")
            print("-" * 50)
            
            # Route to weather forecast subagent with the user's prompt and conversation state
            result_data = run_forecast_agent(user_input, conversation_state)
            
            # Update conversation state for next interaction
            conversation_state = result_data.get("conversation_state")
            
            # Format and display the result
            formatted_output = format_weather_output(result_data)
            print(f"\n🌤️  Assistant: {formatted_output}")
            
        except KeyboardInterrupt:
            print("\n\n👋 Thanks for using Agent Travel Planner! Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try again with a different question.")

if __name__ == "__main__":
    main()