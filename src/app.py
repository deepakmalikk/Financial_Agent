import os
import streamlit as st
from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.yfinance import YFinanceTools
from dotenv import load_dotenv

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

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

def stream_response(query: str, agent: Agent):
    """Streams the agent's response in real time."""
    if not query.strip():
        raise ValueError("Empty query")
    
    try:
        response_generator = agent.run(query, stream=True)
        for chunk in response_generator:
            # Convert chunk to string if it isn't already
            if chunk is not None:
                yield str(chunk)
    except Exception as e:
        yield f"🚨 **Error:** Unable to process the query. \n\n🛠 **Details:** {str(e)}"

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
            st.write("⏳ **Fetching insights...**")
            result_placeholder = st.empty()
            
            try:
                response_text = ""
                for chunk in stream_response(query, team_agent):
                    if chunk:  # Only process non-empty chunks
                        response_text += chunk
                        result_placeholder.markdown(response_text)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
        else:
            st.warning("⚠️ Please enter a query before clicking the button.")

if __name__ == "__main__":
    run_app()
