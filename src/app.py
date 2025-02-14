import os
import streamlit as st
from phi.agent import Agent
from phi.model.groq import Groq
from phi.model.anthropic import Claude  # <-- Make sure you have phi.model.anthropic installed
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.yfinance import YFinanceTools
from dotenv import load_dotenv

load_dotenv()  # Load environment variables if you have a .env file

# Retrieve API keys (if stored in secrets.toml or environment variables)
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY") or st.secrets.get("ANTHROPIC_API_KEY", "")
print(ANTHROPIC_API_KEY)
def create_agents(model_choice: str):
    """
    Create and return the Web Search Agent, Finance Analysis Agent, 
    and Team Agent based on the user's selected model.
    """
    if model_choice == "Groq":
        model = Groq(id="llama-3.3-70b-versatile", api_key=GROQ_API_KEY)
    elif model_choice == "Claude":
        # Adjust the 'id' parameter based on the Claude model you want to use
        model = Claude(id="claude-3-5-sonnet-20240620", api_key=ANTHROPIC_API_KEY)
    else:
        st.warning("Unknown model choice. Defaulting to Groq.")
        model = Groq(id="llama-3.3-70b-versatile", api_key=GROQ_API_KEY)

    # Web Search Agent
    web_search_agent = Agent(
        name="Web_search_agent",
        model=model,
        tools=[DuckDuckGo()],
        instructions=["This agent searches the web for financial news and stock trends."],
        show_tool_calls=True,
        markdown=True
    )

    # Finance Analysis Agent
    finance_agent = Agent(
        name="finance_agent",
        model=model,
        tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True)],
        show_tool_calls=True,
        description="You are an investment analyst specializing in stock prices, fundamentals, and recommendations.",
        instructions=["Format your response using markdown and use tables for better readability."],
        markdown=True
    )

    # Team Agent (combining both web_search_agent and finance_agent)
    team_agent = Agent(
        name="Financer Team",
        model=model,
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
    """
    Handles user queries safely and prevents crashes.
    Returns either the RunResponse or a Streamlit error message.
    """
    if not query.strip():
        raise ValueError("Empty query")

    try:
        result = agent.run(query)
        return result
    except Exception as e:
        return st.error(
            f"🚨 **Error:** Unable to process the query. Please try again later.\n\n"
            f"🛠 **Details:** {str(e)}"
        )

def run_app():
    """
    Run the Streamlit app.
    """
    # Set the page configuration
    st.set_page_config(page_title="Financial Agent", page_icon="📈", layout="wide")

    # Main Title and Introduction
    st.title("📈 Financial Agent")
    st.markdown("""
    Welcome to the Financial Agent app!  
    This tool provides financial insights by leveraging a team of AI agents.  
    Enter your query below to get started.
    """)

    # Sidebar: Model selection
    st.sidebar.header("Model Selection")
    model_choice = st.sidebar.selectbox(
        "Choose a model:",
        ["Groq", "Claude"],
        index=0
    )

    # Sidebar: Examples and instructions
    st.sidebar.header("💡 Query Examples")
    st.sidebar.markdown("""
    - **Tesla stock analysis**  
    - **Apple quarterly earnings**    
    - **Cryptocurrency market news**   
    - **Microsoft stock fundamentals**    
    """)

    st.sidebar.header("📝 How to Use")
    st.sidebar.markdown("""
    1. **Select a model** in the dropdown above.
    2. **Enter your query** in the text box on the main page.
    3. **Click** on **"Get Financial Insights"**.
    4. **Wait** a few moments while our AI agents process your request.
    5. **Review** the detailed results (including data tables and sources) below.
    """)

    # Create the agent team for the chosen model
    _, _, team_agent = create_agents(model_choice)

    # User Query Input
    query = st.text_input(
        "Enter your Query:",
        placeholder="E.g., 'Tesla stock analysis' or 'Latest earnings for Apple'"
    )

    # When the user clicks the button, process the query
    if st.button("Get Financial Insights"):
        if query.strip():
            with st.spinner("⏳ Please wait, our AI agents are thinking..."):
                try:
                    result = process_query(query, team_agent)
                    # If the result is an error widget (from st.error), it won't have a 'content' attribute
                    if hasattr(result, "content"):
                        st.write(result.content)
                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")
        else:
            st.warning("⚠️ Please enter a query before clicking the button.")

if __name__ == "__main__":
    run_app()
