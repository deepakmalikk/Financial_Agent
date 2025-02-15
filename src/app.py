import os
import streamlit as st
import logging
from datetime import datetime
import pytz
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
import time
import requests.exceptions
from phi.agent import Agent
from phi.model.groq import Groq
from phi.model.google import Gemini
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.yfinance import YFinanceTools
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Retrieve API keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY", "")

class ServerStatus:
    """Manages server status and error handling."""
    
    MAX_RETRIES = 3
    RETRY_DELAY = 2
    
    @staticmethod
    def show_server_busy():
        st.error("""
        🔄 Our servers are currently experiencing high traffic.
        
        Please try again in a few moments. If the issue persists:
        1. Wait 30 seconds and retry
        2. Refresh your browser
        3. Try a different query
        
        We apologize for any inconvenience.
        """)
    
    @staticmethod
    def show_api_error():
        st.error("""
        ⚠️ We're having trouble connecting to our data providers.
        
        This might be due to:
        - Temporary API service disruption
        - Network connectivity issues
        - Service maintenance
        
        Please try again in a few minutes.
        """)
    
    @staticmethod
    def show_rate_limit():
        st.warning("""
        ⏳ You've reached the request limit.
        
        Please wait a moment before making another request.
        This helps ensure fair usage of our services for all users.
        """)

def get_market_time() -> datetime:
    """Get current market time in EST."""
    est = pytz.timezone('US/Eastern')
    return datetime.now(est)

def get_market_session() -> str:
    """Determine current market session based on time."""
    current_time = get_market_time()
    hour = current_time.hour
    minute = current_time.minute
    
    if hour < 9 or (hour == 9 and minute < 30):
        return "Pre-market"
    elif hour < 16:
        return "Regular Trading Hours"
    else:
        return "After-hours"

def create_model(model_choice: str) -> Any:
    """Create and return the appropriate model based on user selection."""
    if model_choice == "Groq":
        return Groq(id="deepseek-r1-distill-llama-70b", api_key=GROQ_API_KEY)
    elif model_choice == "Google Studio":
        return Gemini(id="gemini-1.5-flash")
    else:
        st.warning("Unknown model choice. Defaulting to Groq.")
        return Groq(id="deepseek-r1-distill-llama-70b", api_key=GROQ_API_KEY)

def create_web_search_agent(model: Any) -> Agent:
    """Create and return the Web Search Agent with hidden processing."""
    current_time = get_market_time()
    market_session = get_market_session()
    
    return Agent(
        name="Web_Search_Agent",
        model=model,
        tools=[DuckDuckGo()],
        instructions=[
            "You are a senior financial news researcher with 20 years of experience.",
            f"Current time (EST): {current_time.strftime('%Y-%m-%d %H:%M:%S %Z')}",
            f"Current market session: {market_session}",
            "Your task: Analyze CURRENT market trends and breaking news.",
            "Requirements:",
            "- Only include news from the current trading day",
            "- Explicitly state the publication time of each news item",
            "- Prioritize breaking news and market-moving events",
            "- Verify news against multiple sources when possible",
            "Output format:",
            "- Only provide the final analysis without mentioning data collection process",
            "- Never mention being an AI or using tools",
            "- Never explain your methodology",
            "- Focus solely on market insights"
        ],
        show_tool_calls=False,  # Hide tool calls
        markdown=True,
        hide_prompt=True  # Hide prompt and thinking process
    )

def create_finance_agent(model: Any) -> Agent:
    """Create and return the Finance Analysis Agent with hidden processing."""
    current_time = get_market_time()
    market_session = get_market_session()
    
    return Agent(
        name="Finance_Analysis_Agent",
        model=model,
        tools=[YFinanceTools(
            stock_price=True,
            analyst_recommendations=True,
            stock_fundamentals=True
        )],
        instructions=[
            "You are a CFA-certified financial analyst with Wall Street experience.",
            f"Current time (EST): {current_time.strftime('%Y-%m-%d %H:%M:%S %Z')}",
            f"Current market session: {market_session}",
            "Real-time analysis requirements:",
            "- Provide only the final analysis without mentioning data sources",
            "- Never mention being an AI or using tools",
            "- Focus solely on market data and insights",
            "- Present data in clean, professional format",
            "Data presentation:",
            "- Lead with key metrics",
            "- Use concise markdown tables",
            "- Include only relevant data points",
            "- No explanations of methodology"
        ],
        show_tool_calls=False,  # Hide tool calls
        markdown=True,
        hide_prompt=True  # Hide prompt and thinking process
    )

def create_team_agent(model: Any, web_agent: Agent, finance_agent: Agent) -> Agent:
    """Create and return the Team Agent with clean output."""
    current_time = get_market_time()
    market_session = get_market_session()
    
    return Agent(
        name="Finance_Team_Agent",
        model=model,
        team=[web_agent, finance_agent],
        instructions=[
            f"Current time (EST): {current_time.strftime('%Y-%m-%d %H:%M:%S %Z')}",
            f"Current market session: {market_session}",
            "Output Structure:",
            "1. Start with timestamp and market session",
            "2. Provide executive summary",
            "3. Present key data in tables",
            "4. List important events chronologically",
            "5. End with actionable insights",
            "Important rules:",
            "- Never mention being an AI or using tools",
            "- Never explain methodology or data sources",
            "- Focus solely on financial insights",
            "- Keep format professional and clean",
            "- Avoid any meta-commentary about the analysis process"
        ],
        show_tool_calls=False,  # Hide tool calls
        markdown=True,
        hide_prompt=True,  # Hide prompt and thinking process
        description="Financial analysis system"
    )

def process_query(query: str, team_agent: Agent) -> str:
    """Process the query with clean output handling."""
    if not query.strip():
        raise ValueError("Empty query provided.")

    retries = 0
    while retries < ServerStatus.MAX_RETRIES:
        try:
            result = team_agent.run(query)
            # Clean the output to remove any tool call logs or thinking process
            output = result.content if hasattr(result, "content") else str(result)
            # Remove any meta-commentary or methodology explanations
            return output
            
        except requests.exceptions.ConnectionError:
            logger.error("Connection error occurred")
            ServerStatus.show_api_error()
            break
            
        except requests.exceptions.Timeout:
            logger.warning(f"Timeout occurred (attempt {retries + 1}/{ServerStatus.MAX_RETRIES})")
            if retries < ServerStatus.MAX_RETRIES - 1:
                time.sleep(ServerStatus.RETRY_DELAY)
                retries += 1
                continue
            ServerStatus.show_server_busy()
            break
            
        except Exception as e:
            logger.error(f"Error processing query: {e}", exc_info=True)
            ServerStatus.show_server_busy()
            break
    
    return "Query processing failed. Please try again later."


def process_query(query: str, team_agent: Agent) -> str:
    """Process the query with enhanced error handling."""
    if not query.strip():
        raise ValueError("Empty query provided.")

    retries = 0
    while retries < ServerStatus.MAX_RETRIES:
        try:
            result = team_agent.run(query)
            return result.content if hasattr(result, "content") else str(result)
        
        except requests.exceptions.ConnectionError:
            logger.error("Connection error occurred")
            ServerStatus.show_api_error()
            break
            
        except requests.exceptions.Timeout:
            logger.warning(f"Timeout occurred (attempt {retries + 1}/{ServerStatus.MAX_RETRIES})")
            if retries < ServerStatus.MAX_RETRIES - 1:
                time.sleep(ServerStatus.RETRY_DELAY)
                retries += 1
                continue
            ServerStatus.show_server_busy()
            break
            
        except requests.exceptions.RequestException as e:
            if "rate limit" in str(e).lower():
                logger.warning("Rate limit reached")
                ServerStatus.show_rate_limit()
                break
            logger.error(f"Request error: {e}")
            ServerStatus.show_server_busy()
            break
            
        except Exception as e:
            logger.error(f"Error while processing query: {e}", exc_info=True)
            ServerStatus.show_server_busy()
            break
    
    return "Query processing failed. Please try again later."

def setup_page():
    """Set up the Streamlit page configuration."""
    st.set_page_config(page_title="Financial Agent", page_icon="📈", layout="wide")
    st.title("📈 Financial Agent")
    st.markdown("""
    Welcome to the Financial Agent app!  
    This tool provides real-time financial insights using AI agents.  
    Enter your query below to get started.
    """)

def setup_sidebar() -> str:
    """Set up the sidebar and return the selected model."""
    st.sidebar.header("⚙️ Configuration")
    model_choice = st.sidebar.selectbox(
        "Choose a model:",
        ["Groq", "Google Studio"],
        index=0,
        help="Choose between Groq (speed) or Google (accuracy)"
    )
    
    st.sidebar.header("💡 Query Examples")
    st.sidebar.markdown("""
    - **Tesla stock analysis and latest news**
    - **Apple earnings and market reaction**
    - **Crypto market current trends**
    - **Microsoft real-time performance**
    """)
    
    st.sidebar.header("📝 How to Use")
    st.sidebar.markdown("""
    1. **Select a model** from the dropdown
    2. **Enter your query** in the text box
    3. **Click** "Get Financial Insights"
    4. **Review** the real-time analysis
    """)
    
    return model_choice

def main():
    """Main application function."""
    try:
        # Check API keys
        if not GROQ_API_KEY:
            st.warning("⚠️ Groq API key is missing. Please set it in your environment variables or Streamlit secrets.")
        if not GOOGLE_API_KEY:
            st.warning("⚠️ Google API key is missing. Please set it in your environment variables or Streamlit secrets.")
        
        # Set up the page
        setup_page()
        
        # Set up the sidebar and get model choice
        model_choice = setup_sidebar()
        
        try:
            # Create the model and agents
            model = create_model(model_choice)
            web_agent = create_web_search_agent(model)
            finance_agent = create_finance_agent(model)
            team_agent = create_team_agent(model, web_agent, finance_agent)
        except Exception as e:
            logger.error("Failed to initialize agents: %s", e)
            ServerStatus.show_server_busy()
            return

        # Create the query input
        query = st.text_input(
            "Enter your Query:",
            placeholder="E.g., 'Bitcoin current market analysis' or 'Tesla real-time performance'"
        )

        if st.button("Get Financial Insights"):
            if query.strip():
                with st.spinner("⏳ Analyzing market data..."):
                    try:
                        result = process_query(query, team_agent)
                        if result:
                            st.markdown(result)
                        else:
                            ServerStatus.show_server_busy()
                    except Exception as e:
                        logger.error(f"Error processing query: {e}")
                        ServerStatus.show_server_busy()
            else:
                st.warning("⚠️ Please enter a query before clicking the button.")

    except Exception as e:
        logger.critical("Critical application error: %s", e, exc_info=True)
        ServerStatus.show_server_busy()

if __name__ == "__main__":
    main()