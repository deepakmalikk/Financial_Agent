import os
import streamlit as st
import logging
from datetime import datetime
from dotenv import load_dotenv

from agno.agent import Agent, RunResponse
from agno.models.anthropic import Claude
from agno.models.openai import OpenAIChat
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools
from textwrap import dedent

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
    """
    if model_choice == "claude-3-5-haiku-20241022":
        return Claude(id=model_choice, api_key=ANTHROPIC_API_KEY)
    elif model_choice == "gpt-4o-mini":
        return OpenAIChat(id=model_choice, api_key=OPENAI_API_KEY)
    else:
        # Default to Claude if model_choice is not recognized.
        return Claude(id="claude-3-5-haiku-20241022", api_key=ANTHROPIC_API_KEY)

def create_retrieval_agent(model):
    """
    Create an agent dedicated to retrieving data from DuckDuckGo and YFinance.
    This agent will be used to gather the raw data for the final RAG approach.
    """
    retrieval_agent = Agent(
        name="Retrieval Agent",
        model=model,
        tools=[
            DuckDuckGoTools(search=True, news=True),
            YFinanceTools(
                stock_price=True,
                analyst_recommendations=True,
                stock_fundamentals=True,
                historical_prices=True,
                company_info=True,
                company_news=True,
            ),
        ],
        show_tool_calls=True,
        # Minimal instructions, just to gather data
        instructions=dedent("""\
            You are a retrieval agent. Your job is to use the provided tools to:
            1. Search for relevant news or information from DuckDuckGo.
            2. Gather relevant financial data from YFinance.
            3. Return ONLY the raw data or text you found, without any extra commentary.

            IMPORTANT:
            - Do not invent or summarize. Just provide the raw data or short bullet points.
            - If no data is found, say 'No data found.'.
        """),
        add_datetime_to_instructions=True,
        markdown=False,  # We'll parse or format the final output ourselves
    )
    return retrieval_agent

def create_team_agent(model):
    """
    Create the final agent that produces the structured financial analysis
    based on the retrieved data from the retrieval agent.
    """
    team_agent = Agent(
        name="Team Agent",
        model=model,
        instructions=dedent("""\
            You are the final financial analyst. You have the following data:
            1. News data from DuckDuckGo
            2. Financial data from YFinance

            Your task:
            - Integrate the retrieved data into a cohesive, structured financial analysis.
            - Do NOT add or hallucinate any info not present in the retrieved data.
            - If data is insufficient, state: "Insufficient Data."

            Output structure (in Markdown):
            # [Stock/Topic] Analysis

            ## Executive Summary
            - A short overview of the key points.

            ## Financial Data
            - Summarize relevant metrics, using bullet points or tables.

            ## News Highlights
            - Summarize relevant news headlines, quotes, or short snippets.

            ## Key Takeaways
            - Provide 2-3 bullet points of the most important insights.

            ## Risk Factors
            - If relevant, list possible risks or uncertainties.

            End with: "Market Watch Team, {current_date}"
        """),
        show_tool_calls=False,
        add_datetime_to_instructions=True,
        markdown=True,
    )
    return team_agent

def retrieve_context(query: str, retrieval_agent: Agent) -> str:
    """
    Use the Retrieval Agent to gather raw data about 'query'.
    We pass instructions telling the agent to get news & financial info.
    """
    # The prompt instructs the agent to gather data from the tools
    retrieval_prompt = f"""
    Gather any relevant financial news and data about "{query}".
    Return the raw data below. 
    """
    try:
        response: RunResponse = retrieval_agent.run(retrieval_prompt)
        # The agent will call DuckDuckGoTools and YFinanceTools behind the scenes
        if response.content:
            return response.content
        else:
            return "No data found."
    except Exception as e:
        logger.error("Error retrieving data", exc_info=True)
        return "No data found."

def process_query(query: str, retrieval_agent: Agent, team_agent: Agent) -> str:
    """
    Perform a Retrieval-Augmented Generation approach:
    1. Use retrieval_agent to gather data about the query.
    2. Pass that data to team_agent for final structured output.
    """
    if not query.strip():
        st.warning("Please enter a valid financial query.")
        return ""

    # Step 1: Retrieve context (raw data)
    retrieved_data = retrieve_context(query, retrieval_agent)
    current_date = datetime.now().strftime("%Y-%m-%d")

    # Step 2: Pass the retrieved data to the final (team) agent
    final_prompt = dedent(f"""\
        You have the following retrieved data about "{query}":

        {retrieved_data}

        As of {current_date}, provide a structured financial analysis.
        Remember:
        - Use ONLY the above data (no external info).
        - If insufficient, say so.
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
    
    # Example queries in sidebar
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
        ["claude-3-5-haiku-20241022", "gpt-4o-mini"]
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
                # 1) Create model
                model = create_model(model_choice)

                # 2) Create a retrieval agent
                retrieval_agent = create_retrieval_agent(model)

                # 3) Create the final (team) agent
                team_agent = create_team_agent(model)

                # 4) Process query using RAG approach
                result = process_query(query, retrieval_agent, team_agent)
                
                if result:
                    st.markdown(f"**Result from {team_agent.name}:**")
                    st.markdown(result)
            except Exception as e:
                logger.error("Server error", exc_info=True)
                st.error("Our server is busy right now. Please try again later.")

if __name__ == "__main__":
    main()
