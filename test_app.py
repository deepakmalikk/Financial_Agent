import pytest
from app import process_query, create_agents
from phi.agent import Agent

# A simple dummy result class for testing
class DummyResult:
    def __init__(self, content):
        self.content = content

# A dummy agent to test process_query without external dependencies
class DummyAgent(Agent):
    def run(self, query: str):
        return DummyResult(content=f"Processed: {query}")

def test_process_query_empty_string():
    """
    Ensure that an empty query raises a ValueError.
    """
    dummy_agent = DummyAgent(name="dummy", model=None)
    with pytest.raises(ValueError, match="Empty query"):
        process_query("   ", dummy_agent)

def test_process_query_valid_query():
    """
    Ensure that a valid query is processed correctly.
    """
    dummy_agent = DummyAgent(name="dummy", model=None)
    query = "Test Query"
    result = process_query(query, dummy_agent)
    assert result.content == f"Processed: {query}"

def test_create_agents_configuration():
    """
    Verify that the agents are correctly configured.
    """
    dummy_api_key = "dummy_key"
    web_agent, finance_agent, team_agent = create_agents(dummy_api_key)

    # Check that each agent has the expected name
    assert web_agent.name == "Web_search_agent"
    assert finance_agent.name == "finance_agent"
    assert team_agent.name == "Financer Team"

    # Verify that the team agent includes both sub-agents
    team_names = [agent.name for agent in team_agent.team]
    assert "Web_search_agent" in team_names
    assert "finance_agent" in team_names
