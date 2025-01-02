# Financial Agent

## Overview
The Financial Insights Agent is a multi-functional agent designed to assist users with financial research and decision-making. It utilizes advanced AI models and tools to fetch real-time financial data, perform web searches for the latest news, and analyze stock fundamentals to provide actionable insights.

### Key Features
- **Web Search Integration:** Leverages DuckDuckGo to fetch the latest financial news and other relevant data.
- **Stock Analysis:** Uses `YFinanceTools` for detailed stock analysis, including:
  - Stock prices
  - Analyst recommendations
  - Stock fundamentals
- **Team Agent Architecture:** Combines multiple agents for seamless financial research and comprehensive outputs.
- **User-Friendly Outputs:** Provides results formatted in Markdown with tables for better readability.
- **Source Attribution:** Ensures all outputs include sources for transparency and trustworthiness.

---

## Installation

### Prerequisites
- Python 3.9 or higher
- `phi` framework
- `.env` file containing `GROQ_API_KEY`

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Financial_Agent.git
   cd Financial_Agent
