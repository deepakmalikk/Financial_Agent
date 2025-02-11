# Financial Agent

## Overview
The Financial Insights Agent is a multi-functional AI-powered assistant designed to help users with financial research and decision-making. It integrates advanced AI models and real-time data tools to fetch the latest financial news, analyze stock fundamentals, and provide actionable insights in a user-friendly format.

### Key Features
- **Web Search Integration:** Leverages DuckDuckGo to fetch the latest financial news and relevant data.
- **Stock Analysis:** Uses `YFinanceTools` for detailed stock insights, including:
  - Stock prices
  - Analyst recommendations
  - Stock fundamentals
- **Team-Based Agent Architecture:** Combines multiple agents for seamless financial research and comprehensive outputs.
- **User-Friendly Outputs:** Provides results formatted in Markdown with tables for enhanced readability.
- **Source Attribution:** Ensures all outputs include sources for transparency and trustworthiness.
- **Interactive Streamlit UI:** Enables users to input queries and receive insights dynamically with real-time feedback.
- **Loading Indicator:** Displays a waiting message while processing queries to improve user experience.

---

## Installation

### Prerequisites
- Python 3.9 or higher
- `phi` framework
- `streamlit` for UI
- Ollama installed locally to run the AI models

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/deepakmalikk/Financial_Agent.git
   cd Financial_Agent
   ```
2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Ensure Ollama is installed and running locally.

---

## Usage
To start the Financial Agent UI, run the following command:
```bash
streamlit run app.py
```

### How It Works
1. Enter your financial query in the text input box.
2. Click the **"Get Financial Insights"** button.
3. The AI will process your request and fetch relevant financial data and news.
4. Results will be displayed in a well-structured format with tables and sources.

---

## Contributing
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Commit and push your changes.
4. Submit a pull request.

---

## License
This project is licensed under the MIT License.

---

## Contact
For questions or support, reach out to **deepak164malik@gmail.com**.