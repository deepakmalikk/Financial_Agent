from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.yfinance import YFinanceTools
from dotenv import load_dotenv
import os

load_dotenv()
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



myagent = Agent(
    name="Financer Team",
    model = Groq(id="llama-3.1-70b-versatile"),
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

myagent.print_response("Which stock i should buy for short term profit include dates")
