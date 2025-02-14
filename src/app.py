import os
import streamlit as st
from phi.agent import Agent
from phi.model.groq import Groq
from phi.model.google import Gemini  # Using Google Studio's Gemini model
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.yfinance import YFinanceTools
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from a .env file if available

# Retrieve API keys (if needed)
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY", "")

def create_agents(model_choice: str):
    """
    Create and return the Web_Search_Agent, Finance_Analysis_Agent, 
    and the multi-agent Finance_Team_Agent that combines their outputs.
    """
    # Choose the model based on the user's selection
    if model_choice == "Groq":
        model = Groq(id="deepseek-r1-distill-llama-70b", api_key=GROQ_API_KEY)
    elif model_choice == "Google Studio":
        model = Gemini(id="gemini-1.5-flash")
    else:
        st.warning("Unknown model choice. Defaulting to Groq.")
        model = Groq(id="deepseek-r1-distill-llama-70b", api_key=GROQ_API_KEY)
    
    # Create the Web_Search_Agent: Searches the latest finance news.
    web_search_agent = Agent(
        name="Web_Search_Agent",
        model=model,
        tools=[DuckDuckGo()],
        instructions=[
            "You are a senior financial news researcher with 20 years of experience.",
            "Your task: Analyze recent market trends and news for the user's query." 
            "Search the web for the latest financial news and trends based on the query provided."
            "Return a concise summary of the most relevant news articles.",
            "Focus on: Market-moving events, mergers/acquisitions, regulatory changes",
            "Output requirements:",
            "- Structured bullet points with key findings",
            "- Source credibility assessment",
            "- Impact analysis on relevant sectors",
            "- Never mention your data collection methodology"
        ],
        show_tool_calls=True,
        markdown=True
    )
    
    # Create the Finance_Analysis_Agent: Fetches stock data and fundamentals.
    finance_agent = Agent(
        name="Finance_Analysis_Agent",
        model=model,
        tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True)],
        instructions=[
            "You are a CFA-certified financial analyst with Wall Street experience.",
            "Your task: Provide detailed financial analysis for requested instruments."
            "Fetch the current stock data, including price, market cap, volume, and key financial metrics." 
            "Provide the analysis in markdown format using tables for clarity.",
            "Include in analysis:",
            "- Current pricing and historical comparison",
            "- Key financial ratios (P/E, P/B, EV/EBITDA)",
            "- Analyst consensus and price targets",
            "- Risk assessment and volatility metrics",
            "Present data in professional financial reporting format"
        ],
        show_tool_calls=True,
        markdown=True
    )
    
    # Create the multi-agent Finance_Team_Agent which combines results and presents a final answer.
    team_agent = Agent(
        name="Finance_Team_Agent",
        model=model,
        team=[web_search_agent, finance_agent],
        instructions=[
         "You are the Chief Financial Strategist synthesizing inputs from:",
            "1. News Analyst: Provides market context and qualitative insights",
            "2. Data Analyst: Provides quantitative financial metrics",
            "",
            "Synthesis requirements:",
            "- Combine qualitative and quantitative data into unified analysis",
            "- Highlight 3 key actionable insights",
            "- Create risk/reward assessment matrix",
            "- Generate hypothetical investment scenarios",
            "- Format output using professional wealth management standards",
            "- Include markdown tables and charts when applicable"
        
        ],
        show_tool_calls=True,
        markdown=True,
        description="A multi-agent system that combines news and financial data to produce a complete final analysis."
    )
    
    return web_search_agent, finance_agent, team_agent

def process_query(query: str, team_agent: Agent):
    """
    Process the query using the multi-agent Finance_Team_Agent.
    The team agent automatically calls its sub-agents and combines their outputs to generate the final analysis.
    """
    if not query.strip():
        raise ValueError("Empty query provided.")
    
    result = team_agent.run(query)
    return result.content if hasattr(result, "content") else str(result)

def run_app():
    """
    Run the Streamlit app.
    """
    st.set_page_config(page_title="Financial Agent", page_icon="📈", layout="wide")
    
    st.title("📈 Financial Agent")
    st.markdown("""
    Welcome to the Financial Agent app!  
    This tool provides financial insights by leveraging a team of AI agents.  
    Enter your query below to get started.
    """)
    
    # Sidebar: Model selection and instructions
  
    st.header("Configuration")
        model_choice = st.selectbox(
            "AI Model Provider",
            [MODEL_GROQ, MODEL_GOOGLE],
            index=0,
            help="Choose between Groq (speed) or Google (accuracy)"
        )
        
    st.info("""
        **Sample Queries:**
        - Compare risk profiles of TSLA vs F
        - Emerging markets fintech growth analysis
        - Long-term outlook for renewable energy stocks
        """)
       
    st.sidebar.header("💡 Query Examples")
    st.sidebar.markdown("""
    - **Tesla stock analysis**  
    - **Apple quarterly earnings**  
    - **Cryptocurrency market news**  
    - **Microsoft stock fundamentals**  
    """)
    st.sidebar.header("📝 How to Use")
    st.sidebar.markdown("""
    1. **Select a model** from the dropdown.
    2. **Enter your query** in the text box below.
    3. **Click** on **"Get Financial Insights"**.
    4. **Wait** while our AI agents process your request.
    5. **Review** the final analysis which includes tables and references to charts if applicable.
    """)
    
    # Create the agents (the Finance_Team_Agent already contains the other two as its team)
    _, _, team_agent = create_agents(model_choice)
    
    # User Query Input
    query = st.text_input(
        "Enter your Query:",
        placeholder="E.g., 'Bitcoin and Ethereum analysis' or 'Latest earnings for Tesla'"
    )
    
    if st.button("Get Financial Insights"):
        if query.strip():
            with st.spinner("⏳ Please wait, our AI agents are processing your query..."):
                try:
                    final_output = process_query(query, team_agent)
                    st.markdown(final_output)
                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")
        else:
            st.warning("⚠️ Please enter a query before clicking the button.")

if __name__ == "__main__":
    run_app()
