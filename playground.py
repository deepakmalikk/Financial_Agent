from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.yfinance import YFinanceTools
from dotenv import load_dotenv
import os
import phi.api

from phi.playground import Playground, serve_playground_app

load_dotenv()
phi.api = os.getenv("PHI_API_KEY")  
api_key = os.getenv("GROQ_API_KEY")

Web_search_agent = Agent(
    name = "Web_search_agent",
    model = Groq(id="llama-3.1-70b-versatile"),
    tools=[DuckDuckGo()], 
    instructions = ["This agent searches the web for information and also provide sorce."],
    show_tool_calls=True, 
    markdown=True
    )

finance_agent = Agent(
    name = "finance_agent",
    model = Groq(id="llama-3.1-70b-versatile"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True)],
    show_tool_calls=True,
    description="You are an investment analyst that researches stock prices, analyst recommendations, and stock fundamentals.",
    instructions=["Format your response using markdown and use tables to display data where possible."],
    markdown=True
    )

app = Playground(agents =[Web_search_agent, finance_agent]).get_app()

if __name__== "__main__":
    serve_playground_app("playground:app", reload =True)
    #playground is my file name and :app from where my code starts
    