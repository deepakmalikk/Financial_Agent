import os
import streamlit as st
from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.yfinance import YFinanceTools

# Load API key from Streamlit secrets
groq_api_key = st.secrets["GROQ_API_KEY"]

# Webpage configuration
st.set_page_config(page_title="Financial Agent", page_icon="📈", layout="wide")

st.title("📈 Financial Agent - AI-powered Investment Research")

# Example Queries Section
with st.sidebar:
    st.header("💡 Example Queries")
    st.write("- What are the latest trends in the stock market?")
    st.write("- What is the current price of Tesla (TSLA)?")
    st.write("- Can you analyze Apple Inc.'s (AAPL) financials?")
    st.write("- What do analysts say about Amazon (AMZN)?")
    st.write("- Show me a summary of today's financial news.")

# Agent 1 - Web Search
Web_search_agent = Agent(
    name="Web_search_agent",
    model=Groq(id="llama-3.3-70b-versatile", api_key=groq_api_key),
    tools=[DuckDuckGo()],
    instructions=["This agent searches the web for financial news and market trends."],
    show_tool_calls=True,
    markdown=True
)

# Agent 2 - Finance Analysis
finance_agent = Agent(
    name="finance_agent",
    model=Groq(id="llama-3.3-70b-versatile", api_key=groq_api_key),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True)],
    show_tool_calls=True,
    description="You are an investment analyst that researches stock prices, analyst recommendations, and stock fundamentals.",
    instructions=["Format your response using markdown and use tables to display data where possible."],
    markdown=True
)

# Main Agent Team
myagent = Agent(
    name="Financer Team",
    model=Groq(id="llama-3.3-70b-versatile", api_key=groq_api_key),
    team=[Web_search_agent, finance_agent],
    instructions=[
        "First, search finance news for what the user is asking about.",
        "Then, based on stock analysis, provide the output in a structured table.",
        "Include relevant sources in your response.",
        "Provide a concise summary in a well-formatted table.",
    ],
    show_tool_calls=True,
    markdown=True,
    description="A team of AI agents helping with financial research."
)

# Layout for better UI organization
col1, col2 = st.columns(2)

# User Query Input
with col1:
    query = st.text_input("Enter your Query:", placeholder="e.g. What is the latest news on Nvidia?")
    if st.button("Get Financial Insights"):
        if query.strip():
            with st.spinner("⏳ Analyzing data, please wait..."):
                result = myagent.run(query)
            st.write(result.content)
        else:
            st.warning("⚠️ Please enter a query before clicking the button.")

# Stock Market Summary (Example Data)
with col2:
    st.subheader("📊 Market Overview")
    finance_tools = YFinanceTools(stock_price=True)
    stocks = ["AAPL", "TSLA", "AMZN", "GOOGL", "NVDA"]
    stock_data = finance_tools.get_stock_price(stocks)
    st.write(stock_data)

# Financial News Section
st.subheader("📰 Latest Financial News")
news_agent = Agent(
    name="news_agent",
    model=Groq(id="llama-3.3-70b-versatile", api_key=groq_api_key),
    tools=[DuckDuckGo()],
    instructions=["Fetch the latest financial news headlines."]
)
news_result = news_agent.run("Latest financial news headlines")
st.write(news_result.content)

st.markdown("---")
st.caption("Powered by AI - Phidata, DuckDuckGo, and YFinance Tools")
