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
from textwrap import dedent
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
        return Claude(id=model_choice,api_key=ANTHROPIC_API_KEY)
    elif model_choice == "gpt-4o-mini":
        return OpenAIChat(id=model_choice,api_key=OPENAI_API_KEY)
    else:
        # Default to OpenAIChat if model_choice is not recognized.
        return Claude(id="claude-3-5-haiku-20241022",api_key=ANTHROPIC_API_KEY)

def create_my_agent(model) -> Agent:
    
    web_search_agent = Agent(
        name="Web Search Agent",
        role="Search the web for information",
        model=model,
        tools=[DuckDuckGoTools()],
        instructions=dedent("""\
            You are an experienced web researcher and news analyst! 🔍

            Follow these steps when searching for information:
            1. Start with the most recent and relevant sources
            2. Cross-reference information from multiple sources
            3. Prioritize reputable news outlets and official sources
            4. Always cite your sources with links
            5. Focus on market-moving news and significant developments

            Your style guide:
            - Present information in a clear, journalistic style
            - Use bullet points for key takeaways
            - Include relevant quotes when available
            - Specify the date and time for each piece of news
            - Highlight market sentiment and industry trends
            - End with a brief analysis of the overall narrative
            - Pay special attention to regulatory news, earnings reports, and strategic announcements\
        """),
        show_tool_calls=True,
        add_datetime_to_instructions=True,
        markdown=True,
    )

   
    finance_analysis_agent =Agent(
        name="Finance Agent",
        role="Get financial data",
        model=model,
        tools=
        [
            YFinanceTools
            (
                stock_price=True,
                analyst_recommendations=True,
                stock_fundamentals=True,
                historical_prices=True,
                company_info=True,
                company_news=True,
            )
        ],
        instructions=dedent("""\
            You are a skilled financial analyst with expertise in market data! 📊

            Follow these steps when analyzing financial data:
            1. Start with the latest stock price, trading volume, and daily range
            2. Present detailed analyst recommendations and consensus target prices
            3. Include key metrics: P/E ratio, market cap, 52-week range
            4. Analyze trading patterns and volume trends
            5. Compare performance against relevant sector indices

            Your style guide:
            - Use tables for structured data presentation
            - Include clear headers for each data section
            - Add brief explanations for technical terms
            - Highlight notable changes with emojis (📈 📉)
            - Use bullet points for quick insights
            - Compare current values with historical averages
            - End with a data-driven financial outlook\
        """),
        add_datetime_to_instructions=True,
        show_tool_calls=True,
        markdown=True,
)

    

    
    finance_team_agent = Agent(
        name="Team Agent",
        team=[web_search_agent, finance_analysis_agent],
        model=model,
        instructions=dedent("""\
            You are the lead editor of a prestigious financial news desk!
            which always use Web Search Agent  and Finance Agent tools and 
            coordinate to provide latest updates

            Your role:
            1. Coordinate between the Web Search Agent  and Finance Agent
            2. Combine their findings into a compelling narrative
            3. Ensure all information is properly sourced and verified
            4. Present a balanced view of both news and data
            5. Highlight key risks and opportunities

            Your style guide:
            - Start with an attention-grabbing headline
            - Begin with a powerful executive summary
            - Present financial data first, followed by news context
            - Use clear section breaks between different types of information
            - Include relevant charts or tables when available
            - Add 'Market Sentiment' section with current mood
            - Include a 'Key Takeaways' section at the end
            - End with 'Risk Factors' when appropriate
            - Sign off with 'Market Watch Team' and the current date\
        """),
        add_datetime_to_instructions=True,
        show_tool_calls=True,
        markdown=True,
    )
    return finance_team_agent


def process_query(query: str, team_agent: Agent) -> str:

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

# WebPage setup - UI
def setup_streamlit_ui() -> str:
    """Set up the Streamlit UI and return the selected model choice."""
    st.set_page_config(
        page_title="Financial Agent",
        page_icon="📈",
        layout="wide"
    )
    st.title("📈 Financial Agent")
    st.markdown("Enter your query to receive real-time financial analysis and the latest market updates.")
    
    # example queries section
    st.sidebar.header("Example Queries")
    st.sidebar.markdown("""
    - **What’s the latest news and financial performance of Apple (AAPL)?**  
    - **Analyze the impact of AI developments on NVIDIA’s stock (NVDA)**  
    - **What’s the market saying about Amazon’s (AMZN) latest quarter?**  
    - **Analyze Intel's foundry strategy impact on stock performance**
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
                team_agent = create_my_agent(model)
                result = process_query(query, team_agent)
                
                if result:
                    st.markdown(f"**Result from {team_agent.name}:**")
                    st.markdown(result)
            except Exception as e:
                logger.error("Server error", exc_info=True)
                st.error("Our server is busy right now. Please try again after some time.")

if __name__ == "__main__":
    main()
