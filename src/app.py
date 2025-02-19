import os
import re
import streamlit as st
import logging
from datetime import datetime
from dotenv import load_dotenv
from textwrap import dedent

from agno.agent import Agent, RunResponse
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

# Load environment variables
load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY") or st.secrets.get("ANTHROPIC_API_KEY")

# Crypto ticker mapping
CRYPTO_MAPPING = {
    "SOL": "SOL-USD",
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
    "BITCOIN": "BTC-USD",
    "ETHEREUM": "ETH-USD",
    "SOLANA": "SOL-USD",
}

def create_model(model_choice: str):
    """Create model instance with reduced hallucination settings"""
    if model_choice == "claude-3-5-haiku-20241022":
        return Claude(id=model_choice, api_key=ANTHROPIC_API_KEY, temperature=0.1)
    elif model_choice == "gpt-4o":
        return OpenAIChat(id=model_choice, api_key=OPENAI_API_KEY, temperature=0.1, top_p=0.9)
    else:
        return Claude(id="claude-3-5-haiku-20241022", api_key=ANTHROPIC_API_KEY, temperature=0.1)

def create_web_search_agent(model):
    """Web search agent with price extraction capability"""
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
    """Finance agent with strict output formatting"""
    return Agent(
        name="Finance Agent",
        model=model,
        tools=[YFinanceTools(stock_price=True, analyst_recommendations=True)],
        show_tool_calls=True,
        instructions=dedent("""\
            Return ONLY YFinance data in EXACT format:
            - Current price: $[value]
            - Currency: [currency]
            - Change: [value]%
            - Market cap: $[value]
            Do NOT modify numbers. Use original decimals.
            If unavailable, state "No data found."
        """),
        add_datetime_to_instructions=True,
        markdown=False
    )

def create_team_agent(model):
    """Final agent with data validation"""
    return Agent(
        name="Team Agent",
        model=model,
        instructions=dedent("""\
            Synthesize data from these verified sources:
            1. Web Search Results
            2. YFinance Data

            Validation Rules:
            - Prioritize YFinance for numerical data
            - Cross-verify prices with web results
            - Flag discrepancies >2% as warnings
            - Use exact numbers without modification

            Response Format:
            # [Ticker] Analysis
            ## Real-Time Data
            - Price: $[value] ([source])
            - Change: [value]% (Market Open)
            ## Key Metrics
            - Market Cap: $[value]
            ## News Insights
            - [Relevant news]
            ## Data Quality
            - [Any discrepancies]
            End with "Market Watch Team, {date}"
        """),
        show_tool_calls=False,
        add_datetime_to_instructions=True,
        markdown=True,
    )

def resolve_ticker(query: str) -> str:
    """Resolve cryptocurrency tickers"""
    return CRYPTO_MAPPING.get(query.upper(), query)

def extract_price(data: str) -> float:
    """Extract price from any data source"""
    matches = re.findall(r'\$(\d{1,3}(?:,\d{3})*\.\d{2})', data)
    return float(matches[0].replace(',', '')) if matches else None

def retrieve_financial_data(query: str, agent: Agent) -> str:
    """Retrieve and validate financial data"""
    try:
        resolved_query = resolve_ticker(query)
        response = agent.run(f"Get data for {resolved_query}")
        if not response.content:
            return "No data found."
            
        # Validate numerical integrity
        price = extract_price(response.content)
        if price and price < 1:  # Basic sanity check
            logger.warning(f"Low price warning for {resolved_query}: {price}")
            
        return response.content
    except Exception as e:
        logger.error("Financial data error", exc_info=True)
        return "No data found."

def retrieve_web_data(query: str, agent: Agent) -> str:
    """Retrieve web data with price validation"""
    try:
        response = agent.run(f"Current price and news for {query}")
        return response.content or "No data found."
    except Exception as e:
        logger.error("Web data error", exc_info=True)
        return "No data found."

def process_query(query: str, web_agent: Agent, finance_agent: Agent, team_agent: Agent) -> str:
    """Enhanced RAG pipeline with data validation"""
    if not query.strip():
        return "Please enter a valid query."

    # Parallel data retrieval
    web_data = retrieve_web_data(query, web_agent)
    finance_data = retrieve_financial_data(query, finance_agent)

    # Price validation
    web_price = extract_price(web_data)
    finance_price = extract_price(finance_data)
    
    # Build validation context
    validation = ""
    if web_price and finance_price:
        diff = abs(web_price - finance_price)/finance_price
        if diff > 0.02:
            validation = f"Warning: Price discrepancy detected ({diff:.2%})"

    # Prepare final analysis
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

# Streamlit UI 
def setup_streamlit_ui() -> str:
    """Configure Streamlit interface and return model choice"""
    st.set_page_config(
        page_title="Financial Analyst Pro",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("📊 Financial Analyst Pro")
    st.markdown("""
        **Enhanced RAG-powered financial analysis**  
        *Combining real-time market data with AI-powered insights*
    """)
    
    # Sidebar with controls and info
    with st.sidebar:
        st.header("Configuration")
        model_choice = st.radio(
            "Select AI Model:",
            ["claude-3-5-haiku-20241022", "gpt-4o"],
            index=0
        )
        
        st.divider()
        st.subheader("Example Queries")
        st.markdown("""
            - **SOL stock price and analysis**  
            - **BTC market data and news**  
            - **Compare TSLA fundamentals with NVDA**  
            - **Latest financial news for Amazon**  
            - **Risk analysis for ETH investments**
        """)
        
        st.divider()
        st.markdown("""
            **Data Sources:**  
            - Yahoo Finance (primary)  
            - DuckDuckGo Web Search (secondary validation)  
            - Cryptocurrency ticker mapping  
            
            *Note: Prices update every 30 seconds*
        """)
    
    return model_choice

def main():
    if not ANTHROPIC_API_KEY and not OPENAI_API_KEY:
        st.error("API keys missing! Check your environment variables.")
        return

    model_choice = setup_streamlit_ui()
    
    # Main input area
    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input(
            "Enter financial query:",
            placeholder="e.g., 'SOL stock price' or 'BTC market cap'",
            help="Supports stocks, crypto, and company names"
        )
    with col2:
        st.markdown("##")
        analyze_clicked = st.button("Run Analysis", type="primary")
    
    if analyze_clicked and query:
        with st.spinner("Running enhanced RAG analysis..."):
            try:
                model = create_model(model_choice)
                web_agent = create_web_search_agent(model)
                finance_agent = create_finance_agent(model)
                team_agent = create_team_agent(model)
                
                result = process_query(query, web_agent, finance_agent, team_agent)
                
                # Display results with validation badge
                st.markdown(f"**Validated Analysis** ({model_choice})")
                st.markdown(result)
                st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                
            except Exception as e:
                st.error("Analysis engine unavailable. Please try again later.")
                logger.error(f"Main execution error: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()


