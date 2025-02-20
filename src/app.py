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

# Asset mapping for ticker resolution
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
        You are an experienced web researcher and news analyst! 🔍

            Follow these steps when searching for information:
            1. Start with the most recent and relevant sources
            2. Cross-reference information from multiple sources
            3. Prioritize reputable news outlets and official sources
            4. Always cite your sources with links
            5. Focus on market-moving news and significant developments
    
            Your style guide:
            - Present information in a clear, journalistic style
            - Use bullet points for key takeaways
            - Include relevant quotes when available
            - Specify the date and time for each piece of news
            - Highlight market sentiment and industry trends
            - End with a brief analysis of the overall narrative
            - Pay special attention to regulatory news, earnings reports, and strategic announcements\
                Extract up-to-date financial news and data from reliable sources.
                Return results as clear bullet points.
                If no data is found, state "No data found."\
        """),
        add_datetime_to_instructions=True,
        markdown=True
    )

def create_finance_agent(model):
    return Agent(
        name="Finance Agent",
        model=model,
        tools=[YFinanceTools(stock_price=True, analyst_recommendations=True)],
        show_tool_calls=True,
        instructions=dedent("""\
            You are a seasoned Wall Street analyst with deep expertise in market analysis! 📊
    
            Follow these steps for comprehensive financial analysis:
            1. Market Overview
               - Latest stock price
               - 52-week high and low
            2. Financial Deep Dive
               - Key metrics (P/E, Market Cap, EPS)
            3. Professional Insights
               - Analyst recommendations breakdown
               - Recent rating changes
    
            4. Market Context
               - Industry trends and positioning
               - Competitive analysis
               - Market sentiment indicators
    
            Your reporting style:
            - Begin with an executive summary
            - Use tables for data presentation
            - Include clear section headers
            - Add emoji indicators for trends (📈 📉)
            - Highlight key insights with bullet points
            - Compare metrics to industry averages
            - Include technical term explanations
            - End with a forward-looking analysis
    
            Risk Disclosure:
            - Always highlight potential risk factors
            - Note market uncertainties
            - Mention relevant regulatory concerns
            
            STRICT FORMATTING RULES:
            - For cryptocurrencies: Use format "CRYPTO: [SYMBOL] | PRICE: $[value] | CHANGE: [24h change]%"
            - For stocks: Use format "STOCK: [SYMBOL] | PRICE: $[value] | CHANGE: [24h change]%"
            - Never modify numerical values.
            - On invalid symbols or missing data, return "No valid data found".\
        """),
        add_datetime_to_instructions=True,
        markdown=True
    )

def create_team_agent(model, mode="analysis", model_choice=""):
    if mode == "news":
        instructions = dedent("""\
        You are a skilled financial analyst with expertise in market data! 

        Follow these steps when analyzing financial data:
        1. Start with the latest stock price, trading volume, and daily range
        2. Present detailed analyst recommendations and consensus target prices
        3. Include key metrics: P/E ratio, market cap, 52-week range
        4. Analyze trading patterns and volume trends
        5. Compare performance against relevant sector indices

        Your style guide:
        - Use tables for structured data presentation
        - Include clear headers for each data section
        - Add brief explanations for technical terms
        - Highlight notable changes with emojis (📈 📉)
        - Use bullet points for quick insights
        - Compare current values with historical averages
        - End with a data-driven financial outlook
        
        Synthesize the provided news data into a concise, accurate news summary.
            Response Template:
            # Financial News Summary
            {news_summary}
            Market Watch Team, {date}\
        """)
    else:
        # For GPT-4o, if YFinance data is missing or incomplete, incorporate fallback web data.
        extra_instruction = ""
        if model_choice.lower() == "gpt-4o":
            extra_instruction = "Note: If YFinance data is missing or incomplete, incorporate fallback data from web search."
        instructions = dedent(f"""\
            Synthesize data following these rules:
            1. Use YFinance as the primary source.
            2. Use web data as a fallback if YFinance data is missing or incomplete.
            3. Flag any price discrepancies >2% as warnings.
            4. Never modify numerical values.
            {extra_instruction}
            
            Response Template:
            # {{Ticker}} Analysis
            ## Verified Data
            - Price: ${{price}} (YFinance)
            - 24h Change: {{change}}%
            - Market Cap: ${{market_cap}}
            ## Cross-Verification
            {{web_data_summary}}
            ## Data Quality
            {{warnings}}
            Market Watch Team, {{date}}
        """)
    return Agent(
        name="Team Agent",
        model=model,
        instructions=instructions,
        show_tool_calls=False,
        add_datetime_to_instructions=True,
        markdown=True,
    )

def resolve_ticker(query: str) -> str:
    # Refine ticker resolution: first, try exact mapping; then check if a known asset is mentioned.
    query = query.strip().upper()
    if query in ASSET_MAPPING:
        return ASSET_MAPPING[query]
    for key, value in ASSET_MAPPING.items():
        if key in query:
            return value
    # Fallback: remove non-alphanumeric characters and assume it's the ticker.
    cleaned = re.sub(r'[^A-Z0-9-]', '', query)
    return cleaned

def extract_price(data: str) -> float:
    # Try extracting from crypto format
    crypto_match = re.search(r'CRYPTO: \S+ \| PRICE: \$(\d+[\d,\.]*)', data)
    if crypto_match:
        try:
            return float(crypto_match.group(1).replace(",", ""))
        except:
            pass
    # Try stock format
    stock_match = re.search(r'STOCK: \S+ \| PRICE: \$(\d+[\d,\.]*)', data)
    if stock_match:
        try:
            return float(stock_match.group(1).replace(",", ""))
        except:
            pass
    # Generic price extraction
    price_matches = re.findall(r'\$(\d+[\d,\.]*)', data)
    if price_matches:
        try:
            return max(float(p.replace(",", "")) for p in price_matches)
        except:
            pass
    return None

def retrieve_financial_data(query: str, agent: Agent) -> str:
    resolved_symbol = resolve_ticker(query)
    try:
        response = agent.run(f"Get price for {resolved_symbol}")
        if "PRICE: $" not in response.content:
            raise ValueError("Invalid price format")
        return response.content
    except Exception as e:
        logger.error(f"YFinance retrieval failed for {query}: {str(e)}")
        return "No valid data found"

def retrieve_web_data(query: str, agent: Agent) -> str:
    try:
        response = agent.run(f"Extract news and financial data for {query}")
        return response.content or "No data found."
    except Exception as e:
        logger.error("Web data error", exc_info=True)
        return "No data found."

def process_query(query: str, web_agent: Agent, finance_agent: Agent, team_agent_analysis: Agent, team_agent_news: Agent) -> str:
    if not query.strip():
        return "Please enter a valid query."
    
    query_lower = query.lower()
    # Classify query type: if it contains news-related keywords, use news mode.
    if any(keyword in query_lower for keyword in ["news", "trends", "headlines"]):
        web_data = retrieve_web_data(query, web_agent)
        context = f"News Data: {web_data}\nCurrent Date: {datetime.now().strftime('%Y-%m-%d')}"
        try:
            result = team_agent_news.run(context)
            return result.content
        except Exception as e:
            logger.error("News analysis error", exc_info=True)
            return "News analysis unavailable."
    else:
        # For ticker-based financial analysis:
        finance_data = retrieve_financial_data(query, finance_agent)
        # Fallback: if YFinance returns no valid data, use web search data instead.
        if "No valid data found" in finance_data:
            web_data_fallback = retrieve_web_data(query, web_agent)
            finance_data = web_data_fallback
        
        # Attempt to extract prices from the chosen finance data and an extra web lookup.
        web_price = extract_price(finance_data)
        web_data_extra = retrieve_web_data(query, web_agent)
        web_price_extra = extract_price(web_data_extra)
        
        validation = ""
        if web_price and web_price_extra:
            diff = abs(web_price - web_price_extra) / web_price if web_price != 0 else 0
            if diff > 0.02:
                validation = f"Warning: Price discrepancy detected ({diff:.2%})"
        
        context = f"Web Data: {web_data_extra}\nFinance Data: {finance_data}\nValidation Notes: {validation}\nCurrent Date: {datetime.now().strftime('%Y-%m-%d')}"
        try:
            result = team_agent_analysis.run(context)
            return result.content
        except Exception as e:
            logger.error("Financial analysis error", exc_info=True)
            return "Financial analysis unavailable."

def setup_streamlit_ui() -> str:
    st.set_page_config(page_title="Financial Agent", page_icon="📈", layout="wide")
    st.title("📈 Financial Agent")
    st.markdown("Enter your query to receive up-to-date, accurate financial analysis or news updates.")
    
    with st.sidebar:
        st.header("Configuration")
        model_choice = st.radio("Select AI Model:", ["claude-3-5-haiku-20241022", "gpt-4o"], index=0)
        st.divider()
        st.subheader("Example Queries")
        st.markdown("""
        - **AAPL stock analysis:** Latest stock price and market analysis for Apple Inc.
        - **Bitcoin trends:** Current trends in the cryptocurrency market.
        - **Recent tech sector news:** Up-to-date news headlines on the tech sector.
        """)
        st.divider()
        st.markdown("""
            **Data Sources:**  
            - Yahoo Finance (primary)  
            - DuckDuckGo Web Search (fallback)  
            *Data updates frequently.
        """)
    return model_choice

def main():
    if not (ANTHROPIC_API_KEY or OPENAI_API_KEY):
        st.error("API keys missing! Check your environment variables.")
        return

    model_choice = setup_streamlit_ui()
    query = st.text_input("Enter financial query:", placeholder="e.g., TSLA stock analysis or tech sector news")
    analyze_clicked = st.button("Run Analysis", type="primary")
    
    if analyze_clicked and query:
        with st.spinner("Running analysis..."):
            try:
                model = create_model(model_choice)
                web_agent = create_web_search_agent(model)
                finance_agent = create_finance_agent(model)
                team_agent_analysis = create_team_agent(model, mode="analysis", model_choice=model_choice)
                team_agent_news = create_team_agent(model, mode="news", model_choice=model_choice)
                result = process_query(query, web_agent, finance_agent, team_agent_analysis, team_agent_news)
                st.markdown(f"**Validated Analysis** ({model_choice})")
                st.markdown(result)
                st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            except Exception as e:
                st.error("Analysis engine unavailable. Please try again later.")
                logger.error(f"Main execution error: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()
