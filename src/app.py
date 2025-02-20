import os
import re
import streamlit as st
import logging
from datetime import datetime
from dotenv import load_dotenv
from textwrap import dedent

from agno.agent import Agent
from agno.models.anthropic import Claude
from agno.models.openai import OpenAIChat
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load API keys
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY") or st.secrets.get("ANTHROPIC_API_KEY")

# Crypto/stock mapping with common symbols
ASSET_MAPPING = {
    "SOL": "SOL-USD",
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
    "AAPL": "AAPL",
    "TSLA": "TSLA",
    "NVDA": "NVDA",
}

def create_model(model_choice: str):
    if model_choice == "claude-3-5-haiku-20241022":
        return Claude(id=model_choice, api_key=ANTHROPIC_API_KEY, temperature=0.1)
    elif model_choice == "gpt-4o":
        return OpenAIChat(id=model_choice, api_key=OPENAI_API_KEY, temperature=0.1, top_p=0.9)
    else:
        return Claude(id="claude-3-5-haiku-20241022", api_key=ANTHROPIC_API_KEY, temperature=0.1)

def create_web_search_agent(model):
    return Agent(
        name="Web Search Agent",
        model=model,
        tools=[DuckDuckGoTools(search=True, news=True)],
        show_tool_calls=True,
        instructions=dedent("""\
            Extract raw financial data including current prices from reliable sources.
            Return results in this format:
            - Current price: $[value] (from [source])
            - [News bullet points]
            If no data found, state "No data found."
        """),
        add_datetime_to_instructions=True,
        markdown=False
    )

def create_finance_agent(model):
    return Agent(
        name="Finance Agent",
        model=model,
        tools=[YFinanceTools(stock_price=True, analyst_recommendations=True)],
        show_tool_calls=True,
        instructions=dedent("""\
            STRICT FORMATTING RULES:
            - For cryptocurrencies: Use format "CRYPTO: [SYMBOL] | PRICE: $[value] | CHANGE: [24h change]%"
            - For stocks: Use format "STOCK: [SYMBOL] | PRICE: $[value] | CHANGE: [24h change]%"
            - Never modify numerical values
            Error on invalid symbol
        """),
        add_datetime_to_instructions=True,
        markdown=False
    )

def create_team_agent(model):
    return Agent(
        name="Team Agent",
        model=model,
        instructions=dedent("""\
            Synthesize data following these rules:
            1. Use YFinance as primary source
            2. Only use web data when YFinance unavailable
            3. Flag any price differences >2% as warnings
            4. Never modify numerical values
            
            Response Template:
            # {Ticker} Analysis
            ## Verified Data
            - Price: ${price} (YFinance)
            - 24h Change: {change}%
            - Market Cap: ${market_cap}
            ## Cross-Verification
            {web_data_summary}
            ## Data Quality
            {warnings}
            Market Watch Team, {date}
        """),
        show_tool_calls=False,
        add_datetime_to_instructions=True,
        markdown=True,
    )

def resolve_ticker(query: str) -> str:
    clean_query = re.sub(r'[^A-Z0-9-]', '', query.strip().upper())
    return ASSET_MAPPING.get(clean_query, clean_query)

def extract_price(data: str) -> float:
    crypto_match = re.search(r'CRYPTO: \S+ \| PRICE: \$(\d+\.\d{2})', data)
    if crypto_match:
        return float(crypto_match.group(1))
    stock_match = re.search(r'STOCK: \S+ \| PRICE: \$(\d+\.\d{2})', data)
    if stock_match:
        return float(stock_match.group(1))
    price_matches = re.findall(r'\$(\d+\.\d{2})', data)
    if price_matches:
        return max(map(float, price_matches))
    return None

def retrieve_financial_data(query: str, agent: Agent) -> str:
    try:
        resolved_symbol = resolve_ticker(query)
        response = agent.run(f"Get price for {resolved_symbol}")
        if "PRICE: $" not in response.content:
            raise ValueError("Invalid price format")
        return response.content
    except Exception as e:
        logger.error(f"Data retrieval failed for {query}: {str(e)}")
        return "No valid data found"

def retrieve_web_data(query: str, agent: Agent) -> str:
    try:
        response = agent.run(f"Current price and news for {query}")
        return response.content or "No data found."
    except Exception as e:
        logger.error("Web data error", exc_info=True)
        return "No data found."

def process_query(query: str, web_agent: Agent, finance_agent: Agent, team_agent: Agent) -> str:
    if not query.strip():
        return "Please enter a valid query."

    web_data = retrieve_web_data(query, web_agent)
    finance_data = retrieve_financial_data(query, finance_agent)

    web_price = extract_price(web_data)
    finance_price = extract_price(finance_data)
    
    validation = ""
    if web_price and finance_price:
        diff = abs(web_price - finance_price) / finance_price
        if diff > 0.02:
            validation = f"Warning: Price discrepancy detected ({diff:.2%})"

    context = f"""
    Web Data: {web_data}
    Finance Data: {finance_data}
    Validation Notes: {validation}
    Current Date: {datetime.now().strftime('%Y-%m-%d')}
    """
    
    try:
        result = team_agent.run(context)
        return result.content
    except Exception as e:
        logger.error("Analysis error", exc_info=True)
        return "Analysis unavailable."

def setup_streamlit_ui() -> str:
    st.set_page_config(page_title="Financial Agent", page_icon="📈", layout="wide")
    st.title("📈 Financial Agent")
    st.markdown("Enter your query to receive real-time financial analysis and the latest market updates.")
    
    with st.sidebar:
        st.header("Configuration")
        model_choice = st.radio("Select AI Model:", ["claude-3-5-haiku-20241022", "gpt-4o"], index=0)
        st.divider()
        st.subheader("Example Queries")
        st.markdown("""
        - **AAPL stock analysis**: Latest stock price, analyst recommendations, and market analysis for Apple Inc.
        - **Bitcoin market trends**: Current trends in the cryptocurrency market.
        - **Recent tech sector news**: Latest news impacting the technology sector.
        """)
        st.divider()
        st.markdown("""
            **Data Sources:**  
            - Yahoo Finance (primary)  
            - DuckDuckGo Web Search (secondary)  
            *Note: Prices update every 30 seconds*
        """)
    return model_choice

def main():
    if not (ANTHROPIC_API_KEY or OPENAI_API_KEY):
        st.error("API keys missing! Check your environment variables.")
        return

    model_choice = setup_streamlit_ui()
    query = st.text_input("Enter financial query:", placeholder="e.g., TSLA stock price")
    analyze_clicked = st.button("Run Analysis", type="primary")
    
    if analyze_clicked and query:
        with st.spinner("Running enhanced RAG analysis..."):
            try:
                model = create_model(model_choice)
                web_agent = create_web_search_agent(model)
                finance_agent = create_finance_agent(model)
                team_agent = create_team_agent(model)
                result = process_query(query, web_agent, finance_agent, team_agent)
                st.markdown(f"**Validated Analysis** ({model_choice})")
                st.markdown(result)
                st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            except Exception as e:
                st.error("Analysis engine unavailable. Please try again later.")
                logger.error(f"Main execution error: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()
