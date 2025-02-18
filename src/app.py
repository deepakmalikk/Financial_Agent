import os
import streamlit as st
import logging
import time
import random
from datetime import datetime
import pytz
from dotenv import load_dotenv

from agno.agent import Agent
from agno.models.anthropic import Claude
from agno.models.openai import OpenAIChat
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY") or st.secrets.get("ANTHROPIC_API_KEY")

def create_model(model_choice: str):
    """Create and return an LLM model instance based on the selected option."""
    if model_choice == "claude-3-5-haiku-20241022":
        return Claude(api_key=ANTHROPIC_API_KEY, model=model_choice)
    elif model_choice == "gpt-4o-mini":
        return OpenAIChat(api_key=OPENAI_API_KEY, model=model_choice)
    else:
        # Default to OpenAIChat if model_choice is not recognized.
        return OpenAIChat(api_key=OPENAI_API_KEY, model="gpt-4o-mini")

def create_web_search_agent(model) -> Agent:
    """Create an agent that uses DuckDuckGo to fetch the latest financial news."""
    current_date = datetime.now().strftime("%Y-%m-%d")
    instructions = f"""
    <agent_profile>
        <role>Financial News Analyst</role>
        <current_date>{current_date}</current_date>
        <objective>Fetch and analyze the absolute latest financial news using DuckDuckGo</objective>
    </agent_profile>
    <task>
        Retrieve today’s financial news from reputable sources and provide a brief analysis.
    </task>
    """
    return Agent(
        name="WebSearchAgent",
        role="Fetch latest financial news.",
        model=model,
        tools=[DuckDuckGoTools()],
        instructions=[instructions],
        show_tool_calls=False,
        markdown=True
    )

def create_finance_agent(model) -> Agent:
    """Create an agent that uses YFinance to fetch current market data."""
    current_date = datetime.now().strftime("%Y-%m-%d")
    instructions = f"""
    <agent_profile>
        <role>Financial Data Analyst</role>
        <current_date>{current_date}</current_date>
        <objective>Retrieve the latest market data using YFinance</objective>
    </agent_profile>
    <task>
        Use YFinanceTools to fetch current stock prices, analyst recommendations, and company info based on the query.
    </task>
    """
    return Agent(
        name="FinanceAgent",
        role="Fetch latest market data.",
        model=model,
        tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, company_info=True)],
        instructions=[instructions],
        show_tool_calls=False,
        markdown=True
    )

def create_team_agent(model, web_agent: Agent, finance_agent: Agent) -> Agent:
    """Create a team agent that integrates outputs from both web and finance agents."""
    current_date = datetime.now().strftime("%Y-%m-%d")
   
    instructions = f"""
    <agent_profile>
    <role>Financial Analysis Coordinator</role>
    <current_date>{current_date}</current_date>
    <objective>Provide a comprehensive analysis by integrating the latest financial news and market data.</objective>
    </agent_profile>
    <task>
    Gather the latest financial news and up-to-date market data relevant to the user's query. Synthesize this information into a concise and informative response. Ensure that the analysis is based on the most recent data available.
    </task>
    """

    return Agent(
        name="TeamAgent",
        team=[web_agent, finance_agent],
        model=model,
        instructions=[instructions],
        show_tool_calls=False,
        markdown=True
    )

def process_query(query: str, team_agent: Agent) -> str:
    """
    Process the user's query by:
      1. Immediately handling queries about data availability.
      2. Otherwise, delegating to the TeamAgent.
    """
    lower_query = query.lower().strip()
    # Check if the user is asking about data availability.
    if "until when" in lower_query or "till when" in lower_query:
        return "We have access to up-to-date financial data through YFinance and DuckDuckGo APIs."

    if not query.strip():
        st.warning("Please enter a valid financial query.")
        return ""
    
    current_date = datetime.now().strftime("%Y-%m-%d")
    contextual_query = f"Analyze the following query using the latest financial data as of {current_date}: {query}"
    
    try:
        result = team_agent.run(contextual_query)
        return result.content if hasattr(result, "content") else str(result)
    except Exception as e:
        logger.error("Error processing query", exc_info=True)
        st.error("Our server is busy right now. Please try again after some time.")
        return ""

def setup_streamlit_ui() -> str:
    """Set up the Streamlit UI and return the selected model choice."""
    st.set_page_config(
        page_title="Financial Agent",
        page_icon="📈",
        layout="wide"
    )
    st.title("📈 Financial Agent")
    st.markdown("Enter your query to receive real-time financial analysis and the latest market updates.")
    
    # Add example queries section
    st.sidebar.header("Example Queries")
    st.sidebar.markdown("""
    - **AAPL stock analysis**  
      Get the latest stock price, analyst recommendations, and market analysis for Apple Inc.
    - **Bitcoin market trends**  
      Understand the current trends in the cryptocurrency market.
    - **Recent tech sector news**  
      Fetch the latest news impacting the technology sector.
    """)
    
    model_choice = st.sidebar.radio(
        "Select Analysis Model:",
        ["claude-3-5-haiku-20241022", "gpt-4o-mini"]
    )
    return model_choice

def main():
    if not ANTHROPIC_API_KEY:
        st.error("API key is missing. Please configure your API key.")
        return

    model_choice = setup_streamlit_ui()
    query = st.text_input("Enter your financial query:", placeholder="e.g., 'TSLA stock price'")

    if st.button("Analyze"):
        with st.spinner("Fetching the latest financial updates..."):
            try:
                model = create_model(model_choice)
                web_agent = create_web_search_agent(model)
                finance_agent = create_finance_agent(model)
                team_agent = create_team_agent(model, web_agent, finance_agent)
                result = process_query(query, team_agent)
                
                if result:
                    st.markdown(f"**Result from {team_agent.name}:**")
                    st.markdown(result)
                    utc_now = pytz.utc.localize(datetime.utcnow())
                    ist_now = utc_now.astimezone(pytz.timezone("Asia/Kolkata"))
                    st.caption(f"Analysis completed at: {ist_now.strftime('%Y-%m-%d %H:%M:%S IST')}")
            except Exception as e:
                logger.error("Server error", exc_info=True)
                st.error("Our server is busy right now. Please try again after some time.")

if __name__ == "__main__":
    main()
