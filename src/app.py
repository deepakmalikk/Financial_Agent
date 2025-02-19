import os
import streamlit as st
import logging
from datetime import datetime
from dotenv import load_dotenv
from textwrap import dedent

from agno.agent import Agent, RunResponse
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
    """
    Create and return an LLM model instance based on the selected option.
    Lower temperature and controlled top_p are used to reduce hallucinations.
    """
    if model_choice == "claude-3-5-haiku-20241022":
        return Claude(id=model_choice, api_key=ANTHROPIC_API_KEY)
    elif model_choice == "gpt-4o":
        return OpenAIChat(id=model_choice, api_key=OPENAI_API_KEY, temperature=0.2, top_p=0.9)
    else:
        return Claude(id="claude-3-5-haiku-20241022", api_key=ANTHROPIC_API_KEY)

def create_web_search_agent(model):
    """
    Create an agent dedicated to web searching using DuckDuckGoTools.
    """
    web_search_agent = Agent(
        name="Web Search Agent",
        model=model,
        tools=[DuckDuckGoTools(search=True, news=True)],
        show_tool_calls=True,
        instructions=dedent("""\
            You are a web search agent. Your job is to use DuckDuckGo to find relevant, up-to-date financial news and market developments.
            For the given query, return ONLY the raw data in bullet points or short snippets.
            If no relevant news is found, simply state "No data found."
        """),
        add_datetime_to_instructions=True,
        markdown=False
    )
    return web_search_agent

def create_finance_agent(model):
    """
    Create an agent dedicated to financial data retrieval using YFinanceTools.
    """
    finance_agent = Agent(
        name="Finance Agent",
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
        show_tool_calls=True,
        instructions=dedent("""\
            You are a financial data agent. Your job is to use YFinance to retrieve targeted financial data.
            For the given query (such as a ticker), return the latest stock price, analyst recommendations, and relevant financial metrics in bullet points or tables.
            IMPORTANT: Return numeric values exactly as reported by YFinance. Do NOT perform any scaling or conversion (e.g., if the API returns "170", do not output "1.70").
            If no data is found, state "No data found."
        """),
        add_datetime_to_instructions=True,
        markdown=False
    )
    return finance_agent

def create_team_agent(model):
    """
    Create the final team agent that synthesizes data from both the web search and finance agents
    into a cohesive, structured financial analysis.
    """
    team_agent = Agent(
        name="Team Agent",
        model=model,
        instructions=dedent("""\
            You are the final financial analyst. You have been provided with two sets of retrieved data:
            1. Web Search Data (from DuckDuckGo)
            2. Financial Data (from YFinance)

            Your task:
            - Integrate the two sets of data into a cohesive and structured financial analysis.
            - USE ONLY the provided data. Do NOT invent or add any extra information.
            - PRESERVE numeric values exactly as provided. DO NOT perform any scaling or conversion.
            - If the data is insufficient, state "Insufficient Data."

            Please structure your response in Markdown as follows:
            # [Query] Analysis

            ## Executive Summary
            - Brief overview of the key points.

            ## Financial Data
            - Summarize metrics (bullet points or tables).

            ## News Highlights
            - Summarize key news and market updates.

            ## Key Takeaways
            - List 2-3 essential insights.

            ## Risk Factors
            - List any potential risks if applicable.

            End your analysis with "Market Watch Team, {current_date}"
        """),
        show_tool_calls=False,
        add_datetime_to_instructions=True,
        markdown=True,
    )
    return team_agent

def retrieve_web_data(query: str, web_search_agent: Agent) -> str:
    """
    Use the Web Search Agent to gather raw news data about the query.
    A targeted prompt is used to ensure relevant results.
    """
    targeted_prompt = dedent(f"""\
        Find targeted, up-to-date financial news and market developments about "{query}".
        Return the raw data in bullet points or short snippets.
        If no relevant news is found, state "No data found."
    """)
    try:
        response: RunResponse = web_search_agent.run(targeted_prompt)
        return response.content if response.content else "No data found."
    except Exception as e:
        logger.error("Error retrieving web search data", exc_info=True)
        return "No data found."

def retrieve_financial_data(query: str, finance_agent: Agent) -> str:
    """
    Use the Finance Agent to gather raw financial data about the query using YFinance.
    A targeted prompt is used for more accurate results.
    """
    targeted_prompt = dedent(f"""\
        Retrieve the latest financial data for "{query}".
        Include current stock price, analyst recommendations, and relevant financial metrics.
        IMPORTANT: Return numeric values exactly as reported by YFinance. DO NOT perform any scaling or conversion.
        Return the data in bullet points or tables.
        If no data is found, state "No data found."
    """)
    try:
        response: RunResponse = finance_agent.run(targeted_prompt)
        return response.content if response.content else "No data found."
    except Exception as e:
        logger.error("Error retrieving financial data", exc_info=True)
        return "No data found."

def process_query(query: str, web_search_agent: Agent, finance_agent: Agent, team_agent: Agent) -> str:
    """
    Perform a three-step Retrieval-Augmented Generation (RAG) process:
    1. Retrieve web search data using the Web Search Agent.
    2. Retrieve financial data using the Finance Agent.
    3. Pass both results to the Team Agent for final analysis.
    """
    if not query.strip():
        st.warning("Please enter a valid financial query.")
        return ""

    # Step 1: Retrieve web search data
    web_data = retrieve_web_data(query, web_search_agent)
    # Step 2: Retrieve financial data
    financial_data = retrieve_financial_data(query, finance_agent)
    current_date = datetime.now().strftime("%Y-%m-%d")

    # Combine the retrieved data into a single prompt for the team agent.
    final_prompt = dedent(f"""\
        You have the following retrieved data about "{query}":

        --- Web Search Data ---
        {web_data}

        --- Financial Data ---
        {financial_data}

        As of {current_date}, provide a structured financial analysis.
        Remember:
        - Use ONLY the provided data.
        - PRESERVE numeric values exactly as provided (do NOT scale or modify them).
        - If the data is insufficient, state "Insufficient Data."
    """)
    try:
        result: RunResponse = team_agent.run(final_prompt)
        return result.content if result.content else "No final content produced."
    except Exception as e:
        logger.error("Error generating final response", exc_info=True)
        return "Error: Could not generate final response."

def setup_streamlit_ui() -> str:
    """Set up the Streamlit UI and return the selected model choice."""
    st.set_page_config(
        page_title="Financial Agent",
        page_icon="📈",
        layout="wide"
    )
    st.title("📈 Financial Agent")
    st.markdown("Enter your query to receive real-time, retrieval-augmented financial analysis and the latest market updates.")
    
    # Sidebar with example queries
    st.sidebar.header("Example Queries")
    st.sidebar.markdown("""
    - **What’s the latest news and financial performance of Apple (AAPL)?**  
    - **Analyze the impact of AI developments on NVIDIA’s stock (NVDA)**  
    - **What’s the market saying about Amazon’s (AMZN) latest quarter?**  
    - **Analyze Intel's foundry strategy impact on stock performance**
    - **What is the outlook for SOL's price?**
    """)
    
    model_choice = st.sidebar.radio(
        "Select Analysis Model:",
        ["claude-3-5-haiku-20241022", "gpt-4o"]
    )
    return model_choice

def main():
    if not ANTHROPIC_API_KEY and not OPENAI_API_KEY:
        st.error("API key is missing. Please configure your API key.")
        return

    model_choice = setup_streamlit_ui()
    query = st.text_input("Enter your financial query:", placeholder="e.g., 'TSLA stock price'")

    if st.button("Analyze"):
        with st.spinner("Fetching the latest financial updates..."):
            try:
                # 1) Create model with lower temperature settings for reduced hallucinations.
                model = create_model(model_choice)

                # 2) Create three separate agents.
                web_search_agent = create_web_search_agent(model)
                finance_agent = create_finance_agent(model)
                team_agent = create_team_agent(model)

                # 3) Process the query using the three-step RAG approach.
                result = process_query(query, web_search_agent, finance_agent, team_agent)
                
                if result:
                    st.markdown(f"**Result from {team_agent.name}:**")
                    st.markdown(result)
            except Exception as e:
                logger.error("Server error", exc_info=True)
                st.error("Our server is busy right now. Please try again later.")

if __name__ == "__main__":
    main()
