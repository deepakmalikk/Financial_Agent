import os
import streamlit as st
import logging
import requests.exceptions
from datetime import datetime
from phi.agent import Agent
from phi.model.groq import Groq
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

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY","")
GROQ_API_KEY_TWO = os.getenv("GROQ_API_KEY_TWO") or st.secrets.get("GROQ_API_KEY_TWO","")

class DataValidator:
    """Validates and processes financial data."""
    
    @staticmethod
    def validate_ticker(ticker: str) -> bool:
        """Validate if the ticker symbol is properly formatted."""
        if not ticker:
            return False
        # Basic validation for ticker format
        return ticker.isalnum() and 1 <= len(ticker) <= 5

    @staticmethod
    def validate_timeframe(timeframe: str) -> bool:
        """Validate timeframe for data analysis."""
        valid_timeframes = ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', 'max']
        return timeframe in valid_timeframes

class ErrorHandler:
    """Handles various error scenarios."""
    
    @staticmethod
    def handle_api_error():
        st.error("""
        ⚠️ Data Provider Connection Error
        - Checking alternative data sources
        - Please retry in a few moments
        """)

    @staticmethod
    def handle_rate_limit():
        st.warning("⏳ Request limit reached. Please wait a moment before trying again.")

    @staticmethod
    def handle_invalid_query():
        st.warning("⚠️ Please provide a valid financial query or ticker symbol.")

def create_model(model_choice: str) -> Groq:
    """Create and return the appropriate Groq model."""
    models = {
        "llama-3.3-70b-versatile": (GROQ_API_KEY, "llama-3.3-70b-versatile"),
        "deepseek-r1-distill-llama-70b": (GROQ_API_KEY_TWO, "deepseek-r1-distill-llama-70b")
    }
    
    api_key, model_id = models.get(model_choice, (GROQ_API_KEY_TWO, "deepseek-r1-distill-llama-70b"))
    return Groq(id=model_id, api_key=api_key)

def create_web_search_agent(model) -> Agent:
    """Create Web Search Agent with focused financial news analysis."""
    return Agent(
        name="Web_Search_Agent",
        model=model,
        tools=[DuckDuckGo()],
        instructions=[
            """
            <role>
                You are a financial news analyst focused on real-time market intelligence and breaking news.
            </role>

            <data_requirements>
                - Provide only verified, current financial information
                - Focus on breaking news and market-moving events
                - Include source credibility assessment
            </data_requirements>

            <output_format>
                - Timestamp [ET] for each news item
                - Clear headline and source attribution
                - Impact analysis on relevant markets/assets
                - Verification status of information
            </output_format>

            <prohibited>
                - No AI/model references
                - No training data mentions
                - No speculative content
            </prohibited>
            """
        ],
        show_tool_calls=False,
        markdown=True,
        hide_prompt=True
    )

def create_finance_agent(model) -> Agent:
    """Create Finance Agent with comprehensive market analysis capabilities."""
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
            """
            <role>
                You are a quantitative analyst providing real-time financial data analysis.
            </role>

            <analysis_requirements>
                - Real-time price and volume analysis
                - Key financial metrics and ratios
                - Technical indicator calculations
                - Institutional activity monitoring
            </analysis_requirements>

            <output_format>
                - Current market data in clear tables
                - Technical analysis summary
                - Key performance metrics
                - Risk indicators and unusual activity
            </output_format>

            <data_standards>
                - All data must be current
                - Include confidence levels
                - Flag any data anomalies
                - Note significant deviations
            </data_standards>
            """
        ],
        show_tool_calls=False,
        markdown=True,
        hide_prompt=True
    )

def create_team_agent(model, web_agent: Agent, finance_agent: Agent) -> Agent:
    """Create Team Agent for integrated financial analysis."""
    return Agent(
        name="Finance_Team_Agent",
        model=model,
        team=[web_agent, finance_agent],
        instructions=[
            """
            <role>
                You are a comprehensive financial analysis system providing real-time market insights.
            </role>

            <integration_requirements>
                - Combine market data with news analysis
                - Link price movements to events
                - Identify key market drivers
                - Provide actionable insights
            </integration_requirements>

            <output_structure>
                1. Key Findings
                   - Critical updates
                   - Major market moves
                   - Important developments

                2. Market Analysis
                   - Price/volume data
                   - Technical indicators
                   - Comparative metrics

                3. News Impact
                   - Breaking news analysis
                   - Market reaction assessment
                   - Sentiment indicators

                4. Risk Assessment
                   - Current risk factors
                   - Volatility measures
                   - Unusual patterns
            </output_structure>

            <quality_standards>
                - Real-time data verification
                - Cross-reference all information
                - Clear confidence levels
                - Actionable conclusions
            </quality_standards>
            """
        ],
        show_tool_calls=False,
        markdown=True,
        hide_prompt=True
    )

def process_query(query: str, team_agent: Agent) -> str:
    """Process financial queries with enhanced error handling."""
    if not query.strip():
        ErrorHandler.handle_invalid_query()
        return ""

    try:
        result = team_agent.run(query)
        return result.content if hasattr(result, "content") else str(result)
    except requests.exceptions.ConnectionError:
        ErrorHandler.handle_api_error()
    except requests.exceptions.RequestException as e:
        if "rate limit" in str(e).lower():
            ErrorHandler.handle_rate_limit()
        else:
            logger.error(f"Request error: {e}")
            ErrorHandler.handle_api_error()
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        ErrorHandler.handle_api_error()
    
    return ""

def setup_streamlit_ui():
    """Configure Streamlit UI with enhanced features."""
    st.set_page_config(page_title="Financial Insights", page_icon="📈", layout="wide")
    
    # Main title and description
    st.title("📈 Financial Insights Engine")
    st.markdown("""
    Get real-time financial analysis and market insights. Enter any query about:
    - Stocks and market performance
    - Company analysis and news
    - Economic trends and data
    - Market-moving events
    """)

    # Sidebar configuration
    st.sidebar.header("⚙️ Settings")
    model_choice = st.sidebar.selectbox(
        "Analysis Model:",
        ["llama-3.3-70b-versatile", "deepseek-r1-distill-llama-70b"],
        help="Select the analysis model for your queries"
    )

    # Advanced options
    with st.sidebar.expander("🔧 Advanced Options"):
        st.markdown("""
        - Detailed technical analysis
        - Fundamental metrics
        - Market sentiment analysis
        - Historical comparisons
        """)

    return model_choice

def main():
    """Main application function with enhanced error handling."""
    try:
        if not GROQ_API_KEY:
            st.warning("⚠️ API key missing. Please check configuration.")
            return

        model_choice = setup_streamlit_ui()
        
        query = st.text_input(
            "Enter your financial query:",
            placeholder="Example: 'AAPL stock analysis' or 'Bitcoin market trends'"
        )

        if st.button("Analyze"):
            if query.strip():
                with st.spinner("📊 Analyzing financial data..."):
                    model = create_model(model_choice)
                    web_agent = create_web_search_agent(model)
                    finance_agent = create_finance_agent(model)
                    team_agent = create_team_agent(model, web_agent, finance_agent)
                    
                    result = process_query(query, team_agent)
                    if result:
                        st.markdown(result)
            else:
                ErrorHandler.handle_invalid_query()

    except Exception as e:
        logger.critical(f"Application error: {e}", exc_info=True)
        ErrorHandler.handle_api_error()

if __name__ == "__main__":
    main()