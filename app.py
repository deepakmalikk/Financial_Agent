import os
import streamlit as st
from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.yfinance import YFinanceTools

def create_agents(api_key: str):
    """Create and return configured AI agents."""
    # Web Search Agent
    web_search_agent = Agent(
        name="Web_search_agent",
        model=Groq(id="llama-3.3-70b-versatile", api_key=api_key),
        tools=[DuckDuckGo()],
        instructions=["This agent searches the web for financial news and stock trends."],
        show_tool_calls=True,
        markdown=True
    )

    # Finance Analysis Agent
    finance_agent = Agent(
        name="finance_agent",
        model=Groq(id="llama-3.3-70b-versatile", api_key=api_key),
        tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True)],
        show_tool_calls=True,
        description="You are an investment analyst specializing in stock prices, fundamentals, and recommendations.",
        instructions=["Format your response using markdown and use tables for better readability."],
        markdown=True
    )

    # Team Agent
    team_agent = Agent(
        name="Financer Team",
        model=Groq(id="llama-3.3-70b-versatile", api_key=api_key),
        team=[web_search_agent, finance_agent],
        instructions=[
            "First, search finance news for what the user is asking about.",
            "Then, analyze stock prices, recommendations, and fundamentals.",
            "Provide structured results in markdown tables.",
            "Include the source of the data and a brief summary at the end.",
        ],
        show_tool_calls=True,
        markdown=True,
        description="A team of AI agents assisting with financial research."
    )

    return web_search_agent, finance_agent, team_agent

def process_query(query: str, agent: Agent):
    """Handles user queries safely and prevents crashes."""
    if not query.strip():
        raise ValueError("Empty query")
    try:
        result = agent.run(query)
        return result  # ✅ Return the actual response object
    except Exception as e:
        return st.error(f"🚨 **Error:** Unable to process the query. Please try again later.\n\n🛠 **Details:** {str(e)}")  # ✅ Return an error message using Streamlit
    
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
    - **Cryptocurrency market news**   
    - **Microsoft stock fundamentals**    
                        """ 
                        )

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
                if isinstance(result, str):  # ✅ Handle error messages properly
                    st.error(result)
                else:
                    st.write(result.content)  # ✅ Avoid 'str' object error
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
        else:
            st.warning("⚠️ Please enter a query before clicking the button.")
if __name__ == "__main__":
    run_app()