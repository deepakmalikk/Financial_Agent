import os
import streamlit as st
import logging
from datetime import datetime
import pytz
import time
import requests.exceptions
from phi.agent import Agent
from phi.model.groq import Groq
from phi.model.huggingface import HuggingFaceChat
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.yfinance import YFinanceTools
from dotenv import load_dotenv

# Monkey-patch HuggingFaceChat to fix the missing model_dump attribute issue
original_create_assistant_message = HuggingFaceChat._create_assistant_message

def patched_create_assistant_message(self, response_message):
    assistant_message = original_create_assistant_message(self, response_message)
    try:
        # Try to use model_dump if available
        assistant_message.tool_calls = [t.model_dump() for t in response_message.tool_calls]
    except AttributeError:
        # Fall back to using dict() if model_dump is not available
        assistant_message.tool_calls = [t.dict() for t in response_message.tool_calls]
    return assistant_message

HuggingFaceChat._create_assistant_message = patched_create_assistant_message

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
HF_TOKEN = os.getenv("HF_TOKEN") or st.secrets.get("HF_TOKEN", "")

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

def create_model(model_choice: str):
    """Create and return the appropriate model based on user selection."""
    if model_choice == "Groq":
        return Groq(id="deepseek-r1-distill-llama-70b", api_key=GROQ_API_KEY)
    elif model_choice == "Hugging Face":
        # Updated model ID to one that supports chat completions (text-generation task)
        return HuggingFaceChat(
            id="meta-llama/Llama-2-7b-chat-hf",
            max_tokens=4096,
        )
    else:
        st.warning("Unknown model choice. Defaulting to Groq.")
        return Groq(id="deepseek-r1-distill-llama-70b", api_key=GROQ_API_KEY)

def create_web_search_agent(model) -> Agent:
    """Create and return the Web Search Agent for news analysis."""
    current_time = get_market_time()
    market_session = get_market_session()
    
    return Agent(
        name="Web_Search_Agent",
        model=model,
        tools=[DuckDuckGo()],
        instructions=[
            "You are a seasoned financial news researcher with over 20 years of experience.",
            f"Current time (EST): {current_time.strftime('%Y-%m-%d %H:%M:%S %Z')}",
            f"Current market session: {market_session}",
            "Task: Provide a detailed report on current market trends and breaking financial news.",
            "Requirements:",
            "- Include only news published on the current trading day.",
            "- For each news item, provide the headline, publication time, source, and a brief summary.",
            "- Prioritize market-moving events and verify news across multiple sources.",
            "Output format:",
            "- Present the news items as a markdown-formatted list.",
            "- Do not include placeholders; provide actual data if available."
        ],
        show_tool_calls=False,  # Hide tool calls
        markdown=True,
        hide_prompt=True  # Hide prompt and thinking process
    )

def create_finance_agent(model) -> Agent:
    """Create and return the Finance Analysis Agent for financial data."""
    current_time = get_market_time()
    market_session = get_market_session()
    
    return Agent(
        name="Finance_Analysis_Agent",
        model=model,
        tools=[YFinanceTools(
            stock_price=True,
            analyst_recommendations=True,
            stock_fundamentals=True,
            company_info=True, 
            company_news=True
        )],
        instructions=[
            "You are a CFA-certified financial analyst with extensive Wall Street experience.",
            f"Current time (EST): {current_time.strftime('%Y-%m-%d %H:%M:%S %Z')}",
            f"Current market session: {market_session}",
            "Task: Provide a detailed real-time financial analysis for the given query.",
            "Requirements:",
            "- Include key financial metrics such as current stock price, market capitalization, P/E ratio, and analyst ratings.",
            "- Present data in a clear and professional markdown table format.",
            "- Accompany the table with a concise commentary summarizing the financial data.",
            "Output format:",
            "- A markdown table with key financial metrics.",
            "- A brief, actionable commentary based solely on the data provided.",
            "- Do not include any placeholder text; provide actual data."
        ],
        show_tool_calls=False,  # Hide tool calls
        markdown=True,
        hide_prompt=True  # Hide prompt and thinking process
    )

def create_team_agent(model, web_agent: Agent, finance_agent: Agent) -> Agent:
    """Create and return the Team Agent that combines insights from both sub-agents."""
    current_time = get_market_time()
    market_session = get_market_session()
    
    return Agent(
        name="Finance_Team_Agent",
        model=model,
        team=[web_agent, finance_agent],
        instructions=[
            f"Current time (EST): {current_time.strftime('%Y-%m-%d %H:%M:%S %Z')}",
            f"Current market session: {market_session}",
            "Task: Combine the financial analysis and market news into a comprehensive report.",
            "Output Structure:",
            "1. Start with a timestamp and current market session.",
            "2. Provide an executive summary of the market situation.",
            "3. Present a markdown table of key financial metrics.",
            "4. List significant financial news items in chronological order with publication times and sources.",
            "5. End with actionable insights based on the combined data.",
            "Rules:",
            "- Do not mention being an AI or the use of external tools.",
            "- Do not explain your methodology or data sources.",
            "- Ensure that every section contains actual, data-driven content."
        ],
        show_tool_calls=False,  # Hide tool calls
        markdown=True,
        hide_prompt=True,  # Hide prompt and thinking process
        description="Financial analysis system"
    )

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
        ["Groq", "Hugging Face"],
        index=0,
        help="Choose between Groq or Hugging Face"
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
        if not HF_TOKEN:
            st.warning("⚠️ Hugging Face API key is missing. Please set it in your environment variables or Streamlit secrets.")
        
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
