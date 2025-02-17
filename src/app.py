import os
import streamlit as st
import logging
import time
import random
from datetime import datetime
import pytz
from dotenv import load_dotenv

from agno.agent import Agent
from agno.models.groq import Groq
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
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "")
GROQ_API_KEY_TWO = os.getenv("GROQ_API_KEY_TWO") or st.secrets.get("GROQ_API_KEY_TWO", "")

def create_model(model_choice: str) -> Groq:
    """Create and return a Groq model instance based on the selected option."""
    models = {
        "llama-3.1-8b-instant": (GROQ_API_KEY, "llama-3.1-8b-instant"),
        "llama-3.3-70b-versatile": (GROQ_API_KEY_TWO, "llama-3.3-70b-versatile")
    }
    api_key, model_id = models.get(model_choice, (GROQ_API_KEY_TWO, "llama-3.3-70b-versatile"))
    return Groq(id=model_id, api_key=api_key)

def create_web_search_agent(model: Groq) -> Agent:
    """Create an agent that uses DuckDuckGo to fetch the latest financial news."""
    current_date = datetime.now().strftime("%Y-%m-%d")
    instructions = f"""
    <agent_profile>
        <role>Financial News Analyst</role>
        <current_date>{current_date}</current_date>
        <objective>Fetch and analyze the absolute latest financial news using DuckDuckGo</objective>
    </agent_profile>
    <task>
        Retrieve today's financial news from reputable sources and provide a brief analysis.
        Output only the final summary, without any internal processing details.
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

def create_finance_agent(model: Groq) -> Agent:
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
        Output only the final summary, with no extra internal details.
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

def create_team_agent(model: Groq, web_agent: Agent, finance_agent: Agent) -> Agent:
    """Create a team agent that integrates outputs from both web and finance agents."""
    current_date = datetime.now().strftime("%Y-%m-%d")
    instructions = f"""
    <agent_profile>
        <role>Financial Analysis Coordinator</role>
        <current_date>{current_date}</current_date>
        <objective>Integrate the latest news and market data for comprehensive analysis</objective>
    </agent_profile>
    <task>
        Use WebSearchAgent to fetch the latest financial news and FinanceAgent for up-to-date market data.
        Combine both outputs into a concise final summary analysis.
        **Important:** Provide only the final answer in plain text. Do NOT include any internal tool calls or chain-of-thought details.
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

def format_response(response: str) -> str:
    """
    Clean and format the agent's response by joining fragmented single-character lines
    and preserving proper paragraph breaks.
    """
    lines = response.splitlines()
    formatted_lines = []
    buffer = []
    for line in lines:
        stripped = line.strip()
        # If a line consists of a single character, add it to a temporary buffer.
        if len(stripped) == 1:
            buffer.append(stripped)
        else:
            # If buffer has collected letters, join them to form a word.
            if buffer:
                formatted_lines.append("".join(buffer))
                buffer = []
            formatted_lines.append(stripped)
    # Append any remaining buffered letters.
    if buffer:
        formatted_lines.append("".join(buffer))
    
    # Join lines with a double newline for clear paragraph separation.
    return "\n\n".join(formatted_lines)

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
        response_text = result.content if hasattr(result, "content") else str(result)
        return format_response(response_text)
    except Exception as e:
        logger.error("Error processing query", exc_info=True)
        st.error("Our server is busy right now. Please try again after some time.")
        return ""

def setup_streamlit_ui() -> str:
    """Set up the Streamlit UI and return the selected model choice."""
    st.set_page_config(
        page_title="Financial Insights Engine",
        page_icon="📈",
        layout="wide"
    )
    st.title("📈 Financial Insights Engine")
    st.markdown("Enter your query to receive real-time financial analysis and the latest market updates.")
    
    # Add example queries section
    st.sidebar.header("Example Queries")
    st.sidebar.markdown("""
    - **AAPL stock analysis**  
      Get the latest stock price, analyst recommendations, and market analysis for Apple Inc.
    - **Bitcoin market trends**  
      Understand the current trends and forecasts in the cryptocurrency market.
    - **Recent tech sector news**  
      Fetch the latest news impacting the technology sector.
    - **Till when do you have data?**  
      Check data availability information.
    """)
    
    # Pre-select "llama-3.3-70b-versatile" as the default model (index=1)
    model_choice = st.sidebar.radio(
        "Select Analysis Model:",
        ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"],
        index=1
    )
    return model_choice

def main():
    if not GROQ_API_KEY:
        st.error("API key is missing. Please configure your API key.")
        return

    model_choice = setup_streamlit_ui()
    query = st.text_input("Enter your financial query:", placeholder="e.g., 'AAPL stock analysis'")

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
