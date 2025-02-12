import os
import streamlit as st
from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.yfinance import YFinanceTools

def create_agents(api_key: str):
    """
    Create and return the configured agents.

    Returns:
        tuple: (web_search_agent, finance_agent, team_agent)
    """
    # Define Agent 1 - Web Search
    web_search_agent = Agent(
        name="Web_search_agent",
        model=Groq(id="llama-3.3-70b-versatile", api_key=api_key),
        tools=[DuckDuckGo()],
        instructions=["This agent searches the web for information and also provides sources."],
        show_tool_calls=True,
        markdown=True
    )

    # Define Agent 2 - Finance Analysis
    finance_agent = Agent(
        name="finance_agent",
        model=Groq(id="llama-3.3-70b-versatile", api_key=api_key),
        tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True)],
        show_tool_calls=True,
        description="You are an investment analyst that researches stock prices, analyst recommendations, and stock fundamentals.",
        instructions=["Format your response using markdown and use tables to display data where possible."],
        markdown=True
    )

    # Define the Main Agent Team that coordinates the other agents
    team_agent = Agent(
        name="Financer Team",
        model=Groq(id="llama-3.3-70b-versatile", api_key=api_key),
        team=[web_search_agent, finance_agent],
        instructions=[
            "First, search finance news for what the user is asking about.",
            "Then, based on stock analysis, provide the output to the user in a table.",
            "Important: You must provide the source of the information that will be relevant and good to go.",
            "Finally, provide a thoughtful and engaging summary in a table.",
        ],
        show_tool_calls=True,
        markdown=True,
        description="This is a team of agents that can help you with your financial research."
    )
    return web_search_agent, finance_agent, team_agent

def process_query(query: str, agent: Agent):
    """
    Process the user query using the provided agent.

    Args:
        query (str): The user's query.
        agent (Agent): The agent (or team) to process the query.

    Returns:
        The result returned by the agent's run method.

    Raises:
        ValueError: If the query is empty.
    """
    if not query.strip():
        raise ValueError("Empty query")
    result = agent.run(query)
    return result

def run_app():
    """
    Run the Streamlit app.
    """
    # Load API key from Streamlit secrets
    groq_api_key = st.secrets["GROQ_API_KEY"]

    # Webpage configuration
    st.set_page_config(page_title="Financial Agent", page_icon="📈", layout="wide")

    # Main Title and Introduction
    st.title("📈 Financial Agent")
    st.markdown("""
    Welcome to the Financial Agent app!  
    This tool lets you receive financial insights by leveraging a team of AI agents.  
    Enter your query below to get started.
    """)

    # Sidebar with examples and instructions
    st.sidebar.header("💡 Query Examples")
    st.sidebar.markdown("""
    - **Tesla stock analysis**  
    - **Apple quarterly earnings**  
    - **Google recent financial news**  
    - **Market trends for renewable energy**
    """)
    st.sidebar.header("📝 How to Use")
    st.sidebar.markdown("""
    1. **Enter your query** in the text box on the main page.
    2. **Click** on **"Get Financial Insights"**.
    3. **Wait** a few moments while our AI agents process your request.
    4. **Review** the detailed results (including data tables and sources) that will appear below.
    """)

    # Create the agent team using the provided API key
    _, _, team_agent = create_agents(groq_api_key)

    # User Query Input with a placeholder
    query = st.text_input("Enter your Query:", placeholder="E.g., 'Tesla stock analysis' or 'Latest earnings for Apple'")

    # When the user clicks the button, process the query
    if st.button("Get Financial Insights"):
        if query.strip():
            with st.spinner("⏳ Please wait, our AI agents are thinking..."):
                try:
                    result = process_query(query, team_agent)
                    st.write(result.content)
                except Exception as e:
                    st.error(f"An error occurred: {e}")
        else:
            st.warning("⚠️ Please enter a query before clicking the button.")

if __name__ == "__main__":
    run_app()
