import os
import streamlit as st
import logging
import time
import requests
import requests.exceptions
from datetime import datetime, timedelta
import random  # For jitter
from agno.agent import Agent
from agno.models.groq import Groq
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools
from dotenv import load_dotenv
import yfinance as yf  # Import the yfinance library
import pytz  # Import for timezone conversion

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


class ErrorHandler:
    """Handles error scenarios."""

    @staticmethod
    def handle_api_error(message="Data provider connection error. Please retry in a few moments."):
        """Handles API connection errors."""
        st.error(f"⚠️ {message}")

    @staticmethod
    def handle_rate_limit(message="Request limit reached. Please wait before trying again."):
        """Handles API rate limit errors."""
        st.warning(f"⏳ {message}")

    @staticmethod
    def handle_invalid_query(message="Please provide a valid financial query or ticker symbol."):
        """Handles invalid query errors."""
        st.warning(f"⚠️ {message}")

    @staticmethod
    def handle_ticker_not_found(ticker: str):
        """Handles ticker not found errors."""
        st.error(f"⚠️ Ticker symbol '{ticker}' not found. Please check the symbol and try again.")

def create_model(model_choice: str) -> Groq:
    """Create and return the appropriate Groq model instance."""
    models = {
        "llama-3.1-8b-instant": (GROQ_API_KEY, "llama-3.1-8b-instant"),
        "llama-3.3-70b-versatile": (GROQ_API_KEY_TWO, "llama-3.3-70b-versatile")
    }
    api_key, model_id = models.get(model_choice, (GROQ_API_KEY_TWO, "llama-3.3-70b-versatile"))
    return Groq(id=model_id, api_key=api_key)

def create_web_search_agent(model) -> Agent:
    """Create Web Search Agent for real-time financial news using DuckDuckGo."""
    current_date = datetime.now().strftime("%Y-%m-%d")
    return Agent(
        name="Web_Search_Agent",
        role="Financial News Research Specialist.  Focuses on ANALYZING news and its impact, not just reporting the latest date.",
        model=model,
        tools=[DuckDuckGoTools()],
        instructions=[
            f"""
            <agent_profile>
                <role>Financial News Research Specialist. Focuses on ANALYSIS.</role>
                <current_date>{current_date}</current_date>
                <objective>Deliver accurate, timely financial news and market impact analysis</objective>
            </agent_profile>

            <task>
                Use DuckDuckGo to search for the latest financial news related to the user's query.
                Focus on news from reputable financial sources (e.g., Reuters, Bloomberg, Wall Street Journal).
                Prioritize news articles published within the last 24 hours. The goal is to analyze the news, not just report the most recent date.
            </task>

            <search_parameters>
                <timeframe>
                    <recent_priority>Last 24 hours</recent_priority>
                    <historical_context>Up to 7 days when relevant</historical_context>
                </timeframe>
                <sources>
                    <primary>Financial news websites, market reports</primary>
                    <secondary>Company announcements, regulatory filings</secondary>
                    <excluded>Social media speculation, unofficial sources</excluded>
                </sources>
            </search_parameters>

            <data_processing>
                <verification>
                    <source_check>Multiple reliable sources required</source_check>
                </verification>
                <prioritization>
                    <high>Market-moving events, earnings reports</high>
                    <medium>Industry trends, sector analysis</medium>
                    <low>General market commentary</low>
                </prioritization>
            </data_processing>

            <output_structure>
                <format>
                    <timestamp>Eastern Time (ET)</timestamp>
                    <headline>Concise with source attribution</headline>
                    <impact_analysis>Market/asset specific effects</impact_analysis>
                </format>
                <excluded_content>
                    <item>Speculation or rumors</item>
                    <item>Non-financial news</item>
                    <item>AI/model self-references</item>
                </excluded_content>
            </output_structure>

            <example>
                User Query: "Recent news about AAPL stock"
                Expected Output:
                - Headline: "Apple Announces New Product Line" (Source: Reuters)
                  Timestamp: 2024-05-03 10:00 ET
                  Impact Analysis: Positive impact on AAPL stock due to anticipated increase in revenue.
            </example>
            """
        ],
        show_tool_calls=False,
        markdown=True
    )

def create_finance_agent(model) -> Agent:
    """Create Finance Analysis Agent for quantitative market insights using YFinanceTools."""
    current_date = datetime.now().strftime("%Y-%m-%d")
    return Agent(
        name="Finance_Analysis_Agent",
        role="Quantitative Market Analyst. Focuses on ANALYZING data and trends, not just reporting the latest data point.",
        model=model,
        tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, company_info=True)],
        instructions=[
            f"""
            <agent_profile>
                <role>Quantitative financial analyst specializing in market data and metrics. Focuses on ANALYSIS.</role>
                <current_date>{current_date}</current_date>
                <objective>Provide comprehensive market analysis and financial insights</objective>
            </agent_profile>

            <task>
                Use YFinanceTools to gather financial data related to the user's query.
                Focus on retrieving stock prices, analyst recommendations, and company information.
                Analyze the data to identify key trends and insights.  The goal is to analyze the data, not just report the most recent data point.
            </task>

            <data_collection>
                <market_data>
                    <real_time>
                        <prices>Current trading prices and volumes</prices>
                        <indicators>Technical indicators and patterns</indicators>
                        <volatility>Market volatility measures</volatility>
                    </real_time>
                    <historical>
                        <timeframes>Daily, weekly, monthly comparisons</timeframes>
                        <benchmarks>Sector and market indices</benchmarks>
                    </historical>
                </market_data>
                <fundamental_data>
                    <metrics>Key financial ratios and valuations</metrics>
                    <analysis>Institutional holdings and recommendations</analysis>
                </fundamental_data>

            <analysis_parameters>
                <technical>
                    <primary>Price action, volume analysis, momentum</primary>
                    <secondary>Moving averages, relative strength</secondary>
                </technical>
                <fundamental>
                    <primary>Earnings, revenue, growth metrics</primary>
                    <secondary>Industry comparisons, peer analysis</secondary>
                </fundamental>
            </analysis_parameters>

            <output_format>
                <structure>
                    <data_tables>Clear, concise presentation</data_tables>
                    <metrics>Key financial indicators</metrics>
                    <analysis>Direct query-relevant insights</analysis>
                </structure>
                <quality_standards>
                    <accuracy>Verified current data only</accuracy>
                    <relevance>Query-specific information</relevance>
                    <highlight>Significant deviations and anomalies</highlight>
                </quality_standards>
            </output_format>

            <example>
                User Query: "AAPL stock analysis"
                Expected Output:
                - Current Price: $170.00
                  Analyst Recommendations: Buy
                  Company Information: Apple Inc. is a technology company.
                  Analysis: The stock is currently trading at $170.00 and analysts recommend buying it.
            </example>
            """
        ],
        show_tool_calls=False,
        markdown=True
    )

def create_team_agent(model, web_agent: Agent, finance_agent: Agent) -> Agent:
    """Create Team Agent to integrate results from web search and finance analysis agents."""
    current_date = datetime.now().strftime("%Y-%m-%d")
    return Agent(
        name="Finance_Team_Agent",
        team=[web_agent, finance_agent],
        model=model,
        instructions=[
            f"""
            <agent_profile>
                <role>Integrated financial analysis coordinator</role>
                <current_date>{current_date}</current_date>
                <objective>Coordinate financial analysis using available tools and team members</objective>
            </agent_profile>

            > If the user asks "Until when do you have data?", respond ONLY with: "We have access to up-to-date financial data through YFinance and DuckDuckGo APIs." Do NOT add any extra explanation or context.

            <task>
                Coordinate financial analysis by using the YFinanceTools and DuckDuckGoTools directly.
                1. Use YFinanceTools to gather financial data related to the user's query (e.g., stock prices, analyst recommendations, company information).
                2. Use DuckDuckGoTools to search for relevant financial news related to the user's query.
                3. Combine the financial data and news into a comprehensive analysis.  Present the key findings in a clear and concise manner.
            </task>

            <example>
                User Query: "AAPL stock analysis"
                Expected Output:
                - Current Price: $170.00 (Source: YFinanceTools)
                - Analyst Recommendations: Buy (Source: YFinanceTools)
                - Recent News: Apple Announces New Product Line (Source: Reuters)
                - Analysis: Positive news and analyst recommendations suggest a favorable outlook for AAPL.
            </example>
            """
        ],
        show_tool_calls=False,
        markdown=True
    )

def process_query(query: str, team_agent: Agent) -> str:
    """Process the user's financial query with enhanced error handling and retry logic."""

    # Direct response for data availability questions
    if "until when do you have data" in query.lower():
        return "We have access to up-to-date financial data through YFinance and DuckDuckGo APIs."

    if not query.strip():
        ErrorHandler.handle_invalid_query()
        return ""

    try:
        current_date = datetime.now().strftime("%Y-%m-%d")
        contextualized_query = f"""
        Analyze the following query using available tools (YFinanceTools and DuckDuckGoTools) as of {current_date}:
        {query}

        """

        max_retries = 3
        retry_count = 0
        wait_time = 2  # Initial wait time in seconds

        while retry_count < max_retries:
            try:
                result = team_agent.run(contextualized_query)
                return result.content if hasattr(result, "content") else str(result)
            except requests.exceptions.RequestException as e:
                if "rate limit" in str(e).lower():
                    retry_count += 1
                    if retry_count < max_retries:
                        logger.warning(f"Rate limit hit, attempt {retry_count}/{max_retries}")
                        # Exponential backoff with jitter
                        sleep_time = wait_time * (2 ** (retry_count - 1)) + random.uniform(0, 1)
                        time.sleep(sleep_time)
                        continue
                ErrorHandler.handle_rate_limit()
                return ""  # Return empty string to indicate failure
            except Exception as e:
                logger.error(f"Error processing query: {e}", exc_info=True)
                ErrorHandler.handle_api_error()
                return ""  # Return empty string to indicate failure

    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        ErrorHandler.handle_api_error()
        return ""  # Return empty string on general exception

def setup_streamlit_ui():
    """Configure the Streamlit UI with settings for a smooth user experience."""
    st.set_page_config(
        page_title="Financial Insights Engine",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("📈 Financial Insights Engine")
    st.markdown("""
    Get real-time financial analysis, market data, and breaking news in one place.
    Enter your query regarding stocks, market trends, or economic events below.
    """)

    # Display current date
    current_date = datetime.now().strftime("%Y-%m-%d")
    st.sidebar.markdown(f"**Current Date:** {current_date}")

    st.sidebar.header("⚙️ Settings")
    model_choice = st.sidebar.radio(
        "Select Analysis Model:",
        ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"],
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
            placeholder="Examples: 'AAPL stock analysis', 'Bitcoin market trends', 'Recent news affecting tech sector'"
        )

        if st.button("Analyze", key="analyze_button"):
            if query.strip():


                with st.spinner("📊 Analyzing financial data..."):
                    model = create_model(model_choice)
                    web_agent = create_web_search_agent(model)
                    finance_agent = create_finance_agent(model)
                    team_agent = create_team_agent(model, web_agent, finance_agent)

                    result = process_query(query, team_agent)
                    if result:
                        st.markdown(f"**Analysis Result From: {team_agent.name}**")
                        st.markdown(result)
                        # Convert to IST and display
                        utc_now = pytz.utc.localize(datetime.utcnow())
                        ist_timezone = pytz.timezone("Asia/Kolkata")
                        ist_now = utc_now.astimezone(ist_timezone)
                        st.caption(f"Analysis completed at: {ist_now.strftime('%Y-%m-%d %H:%M:%S IST')}")
            else:
                ErrorHandler.handle_invalid_query()

    except Exception as e:
        logger.critical(f"Application error: {e}", exc_info=True)
        st.error("An unexpected error occurred. Please try again later.")
        ErrorHandler.handle_api_error()

if __name__ == "__main__":
    main()