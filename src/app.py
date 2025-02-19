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
        return Claude(id=model_choice, api_key=ANTHROPIC_API_KEY)
    elif model_choice == "gpt-4o-mini":
        return OpenAIChat(id=model_choice, api_key=OPENAI_API_KEY)
    else:
        # Default to Claude if model_choice is not recognized.
        return Claude(id="claude-3-5-haiku-20241022", api_key=ANTHROPIC_API_KEY)

def create_my_agent(model) -> Agent:
    """Create agents with updated instructions that stress retrieval-based responses."""
    web_search_agent = Agent(
        name="Web Search Agent",
        role="Search the web for information",
        model=model,
        tools=[DuckDuckGoTools()],
        instructions=dedent("""\
            You are an experienced web researcher and news analyst! 🔍
            
            Follow these steps when searching for information:
            1. Retrieve the most recent and relevant sources.
            2. Cross-reference information from multiple reputable outlets.
            3. Always include source links.
            4. Focus on market-moving news and significant developments.
            
            Your style guide:
            - Present information in a clear, journalistic style.
            - Use bullet points for key takeaways.
            - Include relevant quotes, dates, and source links.
            - Provide a concise analysis.
            """),
        show_tool_calls=True,
        add_datetime_to_instructions=True,
        markdown=True,
    )

    finance_analysis_agent = Agent(
        name="Finance Agent",
        role="Fetch and analyze financial data",
        model=model,
        tools=[
            YFinanceTools(
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
            1. Retrieve the latest stock price, trading volume, and daily range.
            2. Provide detailed analyst recommendations and key metrics.
            3. Present comparisons and historical context.
            
            Your style guide:
            - Use tables for structured data presentation.
            - Clearly label each data section.
            - Use bullet points for insights.
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
            You are the lead editor of a prestigious financial news desk.
            Your objective is to produce accurate, retrieval-augmented financial analysis.
            
            IMPORTANT: Do not fabricate any information. Base your analysis solely on the "Retrieved Data" provided below.
            If the retrieved data is insufficient to answer the query, clearly state that additional data is needed.
            
            Your role:
            1. Integrate the retrieved news and financial data.
            2. Provide a clear, structured answer using only the provided data.
            3. Reference the source data where applicable.
            
            Your style guide:
            - Start with an attention-grabbing headline.
            - Include a clear executive summary.
            - Present financial data first, followed by the news context.
            - Use bullet points and tables as needed.
            - End with key takeaways and risk factors, if relevant.
            - Sign off with "Market Watch Team" and the current date.
            """),
        add_datetime_to_instructions=True,
        show_tool_calls=True,
        markdown=True,
    )
    return finance_team_agent

def retrieve_context(query: str) -> str:
    """
    Retrieve up-to-date financial data using the retrieval tools.
    This function calls DuckDuckGo for news and YFinance for financial data,
    then returns a combined context string.
    """
    # Retrieve news data via DuckDuckGo
    try:
        ddg_tool = DuckDuckGoTools()
        news_results = ddg_tool.search(query)
    except Exception as e:
        logger.error("Error retrieving news data", exc_info=True)
        news_results = "No news data retrieved."

    # Retrieve financial data via YFinance
    try:
        yf_tool = YFinanceTools(
            stock_price=True,
            analyst_recommendations=True,
            stock_fundamentals=True,
            historical_prices=True,
            company_info=True,
            company_news=True,
        )
        finance_results = yf_tool.run(query)
    except Exception as e:
        logger.error("Error retrieving financial data", exc_info=True)
        finance_results = "No financial data retrieved."

    combined_context = f"News Data:\n{news_results}\n\nFinancial Data:\n{finance_results}"
    return combined_context

def process_query(query: str, team_agent: Agent) -> str:
    """
    Process the user query using a Retrieval-Augmented Generation (RAG) approach.
    1. Retrieve relevant data using DuckDuckGo and YFinance tools.
    2. Combine the retrieved data with the query in a prompt.
    3. Use the team agent (with LLM assistance) to generate an analysis strictly based on the retrieved info.
    """
    lower_query = query.lower().strip()
    # Special case check.
    if "until when" in lower_query or "till when" in lower_query:
        return "We have access to up-to-date financial data through YFinance and DuckDuckGo APIs."

    if not query.strip():
        st.warning("Please enter a valid financial query.")
        return ""

    # Step 1: Retrieve context (facts) using the retrieval tools.
    retrieved_context = retrieve_context(query)
    current_date = datetime.now().strftime("%Y-%m-%d")

    # Step 2: Create a prompt that forces the LLM to base its output solely on the retrieved data.
    final_prompt = dedent(f"""\
        You are provided with retrieved financial data below.
        IMPORTANT: Do not add any information that is not present in the "Retrieved Data" section.
        
        Query: {query}
        
        Retrieved Data:
        {retrieved_context}
        
        As of {current_date}, please provide a detailed and structured financial analysis based solely on the above data.
        If the retrieved information is insufficient, clearly state that additional data is needed.
        """)
    try:
        result = team_agent.run(final_prompt)
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
    st.markdown("Enter your query to receive real-time, retrieval-augmented financial analysis and the latest market updates.")
    
    # Example queries section in sidebar.
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
