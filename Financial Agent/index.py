from phi.agent import Agent
from phi.model.ollama import Ollama
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.yfinance import YFinanceTools
# from dotenv import load_dotenv
# import os

# load_dotenv()
# api_key = os.getenv("DEEPSEEK_API_KEY")

Web_search_agent = Agent(
    name = "Web_search_agent",
    model=Ollama(id="llama3.1"),
    tools=[DuckDuckGo()], 
    instructions = ["This agent searches the web for information and also provide sorce."],
    show_tool_calls=True, 
    markdown=True
    )

finance_agent = Agent(
    name = "finance_agent",
    model=Ollama(id="llama3.1"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True)],
    show_tool_calls=True,
    description="You are an investment analyst that researches stock prices, analyst recommendations, and stock fundamentals.",
    instructions=["Format your response using markdown and use tables to display data where possible."],
    markdown=True
    )



myagent = Agent(
    name="Financer Team",
    model=Ollama(id="llama3.1"),
    team=[Web_search_agent, finance_agent],
    instructions=[
        "First, search finance news for what the user is asking about.",
        "Then, based on Stock anyalysis provide the output to the user in table .",
        "Important: you must provide the source of the information that will be relvant and good to go.",
        "Finally, provide a thoughtful and engaging summary in table.",
    ],
    show_tool_calls=True,
    markdown=True,
    description="This is a team of agents that can help you with your financial research."
)

myagent.print_response("What is the stock price of tata moters?")
