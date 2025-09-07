# 🌤️ Agent Travel Planner

A Python application that provides weather forecasts for any location using AI-powered natural language processing and the Open-Meteo weather API.

## 🚀 What It Does

This application uses Google's Gemini AI model to understand natural language requests and automatically fetch weather forecasts for any city worldwide. Simply ask for weather information in plain English, and the AI will:

- Parse your request to extract location and date information
- Look up the location coordinates using geocoding
- Fetch real-time weather data from Open-Meteo API
- Present the forecast in a user-friendly format

## ✨ Features

- 🤖 **AI-Powered**: Uses Google Gemini for natural language understanding
- 🌍 **Global Coverage**: Works with any city worldwide
- 📅 **Flexible Dates**: Get forecasts for today or any future date
- 🎯 **Accurate**: Uses Open-Meteo's reliable weather data
- 💻 **Command Line Interface**: Easy to use from terminal
- 🔧 **Easy Setup**: Automated installation scripts included

## 📋 Prerequisites

- Python 3.8 or higher
- Google Gemini API key ([Get one here](https://makersuite.google.com/app/apikey))

## 🛠️ Installation

### Option 1: Automated Setup (Recommended)

**For Unix/macOS:**
```bash
git clone <your-repo-url>
cd agent-travel-planner
./install.sh
```

**For Windows:**
```bash
git clone <your-repo-url>
cd agent-travel-planner
python3 setup.py
```

### Option 2: Manual Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd agent-travel-planner

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Unix/macOS
# or
venv\Scripts\activate     # On Windows

# Install dependencies
pip install -r requirements.txt

# Create environment file
cp .env.template .env
```

## ⚙️ Configuration

1. Edit the `.env` file and add your Google Gemini API key:
   ```
   GEMINI_API_KEY=your_actual_api_key_here
   ```

2. Get your API key from [Google AI Studio](https://makersuite.google.com/app/apikey)

## 🎯 Usage

**Note**: Use `python3` to run the setup script, but use `python` (without the "3") once the virtual environment is activated.

### Basic Usage

```bash
# Get weather for a city today
python main.py "Joinville"

# Get weather for a specific date
python main.py "Berlin" "2025-08-22"

# Get weather for any location
python main.py "New York" "2025-01-15"
```

### Examples

```bash
# Today's weather in Paris
python main.py "Paris"

# Weather in Tokyo on New Year's Day
python main.py "Tokyo" "2025-01-01"

# Weather in São Paulo on a specific date
python main.py "São Paulo" "2025-06-15"
```

### Expected Output

```
🤖 Processing request: What is the weather in Joinville on 2025-01-15?
🔧 Executing tool call...
🌍 Looking up weather for Joinville on 2025-01-15...

🌤️  Weather Forecast for Joinville (-26.30, -48.85)
📅 Date: 2025-01-15
==================================================
🕐 00:00: 22°C
🕐 01:00: 21°C
🕐 02:00: 20°C
🕐 03:00: 19°C
...
```

## 🏗️ Project Structure

```
agent-travel-planner/
├── main.py              # Main application file
├── requirements.txt     # Python dependencies
├── setup.py            # Python installation script
├── install.sh          # Shell installation script
├── .env.template       # Environment variables template
├── .gitignore          # Git ignore rules
├── README.md           # This file
└── venv/               # Virtual environment (created during setup)
```

## 🔧 Dependencies

- **langchain-google-genai**: Google Gemini AI integration
- **geopy**: Geocoding and location services
- **requests**: HTTP requests for weather API
- **python-dotenv**: Environment variable management
- **pydantic**: Data validation

## 🐛 Troubleshooting

### Common Issues

**"Location not found" error:**
- Try using the full city name with country: `"Paris, France"`
- Check spelling of the city name
- Use English city names when possible

**"API key not found" error:**
- Make sure your `.env` file exists and contains `GEMINI_API_KEY=your_key`
- Verify your API key is valid and active

**"No weather data available" error:**
- The date might be too far in the future (Open-Meteo has limits)
- Try a date within the next 7-10 days

### Getting Help

1. Check that your virtual environment is activated
2. Verify all dependencies are installed: `pip list`
3. Test your API key: `python -c "import os; from dotenv import load_dotenv; load_dotenv(); print('API Key:', 'SET' if os.getenv('GEMINI_API_KEY') else 'NOT SET')"`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- [Google Gemini](https://ai.google.dev/) for AI capabilities
- [Open-Meteo](https://open-meteo.com/) for weather data
- [LangChain](https://langchain.com/) for AI framework
- [Geopy](https://geopy.readthedocs.io/) for geocoding services

---

**Happy Weather Forecasting! 🌤️**
