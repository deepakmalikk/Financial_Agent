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

def safe_process_query(query: str, agent: Agent):
    """Handles user queries safely and prevents crashes."""
    if not query.strip():
        raise ValueError("Empty query")
    try:
        return agent.run(query)
    except Exception as e:
        return f"🚨 **Error:** Unable to process the query. Please try again later.\n\n🛠 **Details:** {str(e)}"

def run_app():
    """Run the Streamlit app."""
    # Load API key from secrets
    groq_api_key = st.secrets["GROQ_API_KEY"]

    # Webpage configuration
    st.set_page_config(page_title="Financial AI Agent", page_icon="📈", layout="wide")

    # Title and Intro
    st.title("📈 Financial AI Agent")
    st.markdown("""
    **Your AI-powered assistant for stock market trends, financial insights, and investment research.**  
    Enter a query below to get started!
    """)

    # Sidebar
    with st.sidebar:
        st.header("💡 Example Queries")
        st.write("- Tesla stock analysis")
        st.write("- Apple quarterly earnings")
        st.write("- Market trends for renewable energy")
        st.write("- Cryptocurrency market news")
        st.write("- Microsoft stock fundamentals")
        
        st.header("📝 How to Use")
        st.write("1️⃣ Enter your query in the text box.  "
                 "2️⃣ Click on 'Get Financial Insights'. " 
                " 3️⃣ Wait while our AI agents process your request.  "
                 "4️⃣ Review the results (including data tables and sources).")

    # Create AI Agents
    _, _, team_agent = create_agents(groq_api_key)

    # Layout for UI organization
    col1, col2 = st.columns([3, 2])

    # User Query Input
    with col1:
        query = st.text_input("🔍 Enter your Query:", placeholder="E.g., Tesla stock analysis")
        if st.button("Get Financial Insights"):
            if query.strip():
                with st.spinner("⏳ Analyzing data, please wait..."):
                    result = safe_process_query(query, team_agent)
                    st.write(result.content if hasattr(result, "content") else result)
            else:
                st.warning("⚠️ Please enter a query before clicking the button.")

    # Market Overview Section
    with col2:
        st.subheader("📊 Market Overview")
        try:
            finance_tools = YFinanceTools(stock_price=True)
            stocks = ["AAPL", "TSLA", "AMZN", "GOOGL", "NVDA"]
            stock_data = finance_tools.get_stock_price(stocks)
            st.write(stock_data)
        except Exception:
            st.warning("⚠️ Unable to fetch stock market data. Please try again later.")

    # Financial News Section
    st.subheader("📰 Latest Financial News")
    try:
        news_agent = Agent(
            name="news_agent",
            model=Groq(id="llama-3.3-70b-versatile", api_key=groq_api_key),
            tools=[DuckDuckGo()],
            instructions=["Fetch the latest financial news headlines."]
        )
        news_result = news_agent.run("Latest financial news headlines")
        st.write(news_result.content)
    except Exception:
        st.warning("⚠️ Unable to fetch financial news. Please try again later.")

    # Footer
    st.markdown("---")
    st.caption("🔹 Powered by AI - Phi3, DuckDuckGo, and YFinance Tools")

if __name__ == "__main__":
    run_app()