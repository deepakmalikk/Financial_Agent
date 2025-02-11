from phi.agent import Agent
from phi.model.ollama import Ollama
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.yfinance import YFinanceTools
import streamlit as st

# Webpage configuration
st.set_page_config(
    page_title="Financial Agent",  
    page_icon="📈",  
)

st.title("📈 Financial Agent")

# Agent 1 - Web Search
Web_search_agent = Agent(
    name="Web_search_agent",
    model=Ollama(id="llama3.1"),
    tools=[DuckDuckGo()], 
    instructions=["This agent searches the web for information and also provides sources."],
    show_tool_calls=True, 
    markdown=True
)

# Agent 2 - Finance Analysis
finance_agent = Agent(
    name="finance_agent",
    model=Ollama(id="llama3.1"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True)],
    show_tool_calls=True,
    description="You are an investment analyst that researches stock prices, analyst recommendations, and stock fundamentals.",
    instructions=["Format your response using markdown and use tables to display data where possible."],
    markdown=True
)

# Main Agent Team
myagent = Agent(
    name="Financer Team",
    model=Ollama(id="llama3.1"),
    team=[Web_search_agent, finance_agent],
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

# Getting User Query
query = st.text_input("Enter your Query:")

if st.button("Get Financial Insights"):
    if query.strip():  # Check if query is not empty
        with st.spinner("⏳ Please wait, our AI agents are thinking..."):
            result = myagent.run(query)
        st.write(result.content)
    else:
        st.warning("⚠️ Please enter a query before clicking the button.")