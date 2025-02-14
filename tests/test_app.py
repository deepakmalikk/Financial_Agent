# test_financial_agent.py

import pytest
from src/app import create_agents, process_query  # Replace `your_module` with your actual module name

# --- Dummy Classes for Testing ---

class DummyResult:
    """A dummy result class that mimics the agent's run() output."""
    def __init__(self, content):
        self.content = content

class DummyAgent:
    """A dummy agent with a run() method that returns a DummyResult."""
    def run(self, query):
        return DummyResult(f"Processed query: {query}")

class ExceptionAgent:
    """A dummy agent whose run() method raises an Exception."""
    def run(self, query):
        raise Exception("Test exception")


# --- Pytest Fixtures ---

@pytest.fixture
def dummy_team_agent():
    """Provides a dummy team agent for testing process_query."""
    return DummyAgent()


# --- Test Cases ---

def test_create_agents_groq():
    """
    Test that create_agents returns a valid team agent when "Groq" is chosen.
    """
    web_search_agent, finance_agent, team_agent = create_agents("Groq")
    # Check that the team agent is not None and has a run method
    assert team_agent is not None
    assert hasattr(team_agent, "run")
    # Optionally, check that the instructions list is non-empty
    assert isinstance(team_agent.instructions, list)
    assert len(team_agent.instructions) > 0


def test_create_agents_google():
    """
    Test that create_agents returns a valid team agent when "Google Studio" is chosen.
    """
    web_search_agent, finance_agent, team_agent = create_agents("Google Studio")
    assert team_agent is not None
    assert hasattr(team_agent, "run")
    assert isinstance(team_agent.instructions, list)
    assert len(team_agent.instructions) > 0


def test_create_agents_unknown_model(monkeypatch):
    """
    Test that an unknown model choice defaults to Groq.
    """
    warning_messages = []

    # Monkey-patch st.warning to capture warning messages instead of displaying them.
    monkeypatch.setattr("your_module.st.warning", lambda msg: warning_messages.append(msg))
    
    web_search_agent, finance_agent, team_agent = create_agents("Invalid Model")
    # Verify that a warning was issued about the unknown model choice.
    assert any("Unknown model choice" in message for message in warning_messages)
    assert team_agent is not None
    assert hasattr(team_agent, "run")


def test_process_query_success(dummy_team_agent):
    """
    Test that process_query returns the expected content when a valid query is provided.
    """
    query = "Test Query"
    output = process_query(query, dummy_team_agent)
    assert "Processed query: Test Query" in output


def test_process_query_empty_query(dummy_team_agent):
    """
    Test that providing an empty (or whitespace-only) query raises a ValueError.
    """
    with pytest.raises(ValueError):
        process_query("    ", dummy_team_agent)


def test_process_query_exception():
    """
    Test that if the agent's run() method raises an exception, process_query 
    wraps it and raises a RuntimeError.
    """
    with pytest.raises(RuntimeError):
        process_query("Some query", ExceptionAgent())
