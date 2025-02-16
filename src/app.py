import os
import streamlit as st
import logging
import requests.exceptions
from datetime import datetime
from agno.agent import Agent
from agno.models.groq import Groq
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools
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
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "")
GROQ_API_KEY_TWO = os.getenv("GROQ_API_KEY_TWO") or st.secrets.get("GROQ_API_KEY_TWO", "")

class DataValidator:
    """Validates and processes financial data."""
    
    @staticmethod
    def validate_ticker(ticker: str) -> bool:
        """Validate if the ticker symbol is properly formatted."""
        return bool(ticker) and ticker.isalnum() and 1 <= len(ticker) <= 5

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
        - Checking alternative data sources...
        - Please retry in a few moments.
        """)

    @staticmethod
    def handle_rate_limit():
        st.warning("⏳ Request limit reached. Please wait before trying again.")

    @staticmethod
    def handle_invalid_query():
        st.warning("⚠️ Please provide a valid financial query or ticker symbol.")

def create_model(model_choice: str) -> Groq:
    """Create and return the appropriate Groq model instance."""
    models = {
        "llama-3.1-8b-instant": (GROQ_API_KEY, "llama-3.1-8b-instant"),
        "mixtral-8x7b-32768": (GROQ_API_KEY_TWO, "mixtral-8x7b-32768")
    }
    api_key, model_id = models.get(model_choice, (GROQ_API_KEY_TWO, "mixtral-8x7b-32768"))
    return Groq(id=model_id, api_key=api_key)

def create_web_search_agent(model) -> Agent:
    """Create Web Search Agent for real-time financial news using DuckDuckGo."""
    return Agent(
        name="Web_Search_Agent",
        role="Search the web for information",
        model=model,
        tools=[DuckDuckGoTools()],
        instructions=[
            """
            <role>
                You are a financial news analyst focused on real-time market intelligence.
            </role>

            <data_requirements>
                - Provide only verified, current financial news and market events.
                - Focus on breaking news that impacts financial markets.
            </data_requirements>

            <output_format>
                - Include a timestamp [ET] for each news item.
                - Provide a concise headline with source attribution.
                - Analyze the impact on relevant markets or assets.
                - Do not include extra commentary beyond the query.
            </output_format>

            <prohibited>
                - No speculative or off-topic content.
                - No AI/model meta-references.
            </prohibited>
            """
        ],
        show_tool_calls=True,
        markdown=True,
        hide_prompt=True
    )

def create_finance_agent(model) -> Agent:
    """Create Finance Analysis Agent for quantitative market insights using YFinanceTools."""
    return Agent(
        name="Finance_Analysis_Agent",
        role="Get financial data",
        model=model,
        tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, company_info=True)],
        instructions=[
            """
            <role>
                You are a quantitative analyst providing real-time market data and financial metrics.
            </role>

            <analysis_requirements>
                - Retrieve current stock price and volume data.
                - Compute key financial ratios and technical indicators.
                - Monitor institutional activity and unusual market moves.
            </analysis_requirements>

            <output_format>
                - Present data in clear, concise tables.
                - Provide only information directly answering the user's query.
                - Avoid extraneous details.
            </output_format>

            <data_standards>
                - All data must be current and verified.
                - Highlight any anomalies or significant deviations.
            </data_standards>
            """
        ],
        show_tool_calls=True,
        markdown=True,
        hide_prompt=True
    )

def create_team_agent(model, web_agent: Agent, finance_agent: Agent) -> Agent:
    """Create Team Agent to integrate results from web search and finance analysis agents."""
    return Agent(
        name="Finance_Team_Agent",
        team=[web_agent, finance_agent],
        model=model,
        instructions=[
            """
            <role>
                You are an integrated financial analysis system consolidating real-time news and market data.
            </role>

            <integration_requirements>
                - Combine insights from web search and finance analysis.
                - Cross-reference price movements with market events.
                - Provide a direct and concise answer strictly based on the user's query.
            </integration_requirements>

            <output_structure>
                1. Key Findings
                   - Critical updates and market moves directly related to the query.
                2. Market Analysis
                   - Current price/volume data, key metrics, and technical indicators.
                3. News Impact
                   - Relevant breaking news and sentiment analysis.
            </output_structure>

            <quality_standards>
                - Only include information that directly answers the query.
                - Omit any extraneous commentary.
                - Provide clear source attributions and confidence levels.
            </quality_standards>
            """
        ],
        show_tool_calls=True,
        markdown=True,
        hide_prompt=True,
        debug_mode=True
    )

def process_query(query: str, team_agent: Agent) -> str:
    """Process the user's financial query with enhanced error handling."""
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
    """Configure the Streamlit UI with settings for a smooth user experience."""
    st.set_page_config(page_title="Financial Insights Engine", page_icon="📈", layout="wide")
    
    st.title("📈 Financial Insights Engine")
    st.markdown("""
    Get real-time financial analysis, market data, and breaking news in one place.
    Enter your query regarding stocks, market trends, or economic events below.
    """)

    st.sidebar.header("⚙️ Settings")
    model_choice = st.sidebar.selectbox(
        "Select Analysis Model:",
        ["llama-3.1-8b-instant", "mixtral-8x7b-32768"],
        help="Choose the model powering the financial analysis."
    )

    with st.sidebar.expander("🔧 Advanced Options"):
        st.markdown("""
        - Detailed technical analysis
        - Fundamental metrics
        - Market sentiment tracking
        - Historical comparisons
        """)

    return model_choice

def main():
    """Main application function orchestrating the multiagent framework."""
    try:
        if not GROQ_API_KEY:
            st.warning("⚠️ API key missing. Please check your configuration.")
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
