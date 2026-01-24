import pytest
import pandas as pd
from unittest.mock import Mock, MagicMock
from src.data_generator import DataGenerator

# Mock LLM Client to avoid real API calls and ensure deterministic tests
class MockLLMClient:
    def __init__(self, response_text):
        self.response_text = response_text
        self.generate_called_with = None

    def generate(self, prompt, system_prompt, temperature, max_tokens):
        self.generate_called_with = {
            "prompt": prompt,
            "system_prompt": system_prompt,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        return self.response_text

@pytest.fixture
def mock_client():
    # Return a client that returns valid JSON by default
    return MockLLMClient('{"name": ["Alice", "Bob"], "age": [25, 30]}')

@pytest.fixture
def generator(mock_client):
    return DataGenerator(mock_client)

def test_build_user_query(generator):
    instructions = "Create users"
    n_rows = 5
    query = generator.build_user_query(instructions, n_rows)
    assert "EXACTLY 5 rows" in query
    assert instructions in query

def test_generate_features_basic(generator, mock_client):
    # Setup
    instructions = "Get names and ages"
    n_rows = 2
    existing_df = pd.DataFrame()
    
    # Execute
    df = generator.generate_features(instructions, n_rows, existing_df)
    
    # Verify
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert "name" in df.columns
    assert "age" in df.columns
    assert df.iloc[0]["name"] == "Alice"
    
    # Verify LLM was called correctly
    assert mock_client.generate_called_with["temperature"] == 0.7

def test_generate_features_append(generator, mock_client):
    # Setup: Mock client returns data for NEW columns
    mock_client.response_text = '{"city": ["NY", "LA"]}'
    
    # Existing dataframe
    existing_df = pd.DataFrame({"name": ["Alice", "Bob"]})
    
    # Execute
    df = generator.generate_features("Add city", 2, existing_df)
    
    # Verify
    assert len(df) == 2
    assert len(df.columns) == 2 # name + city
    assert "name" in df.columns
    assert "city" in df.columns
    assert df.iloc[0]["city"] == "NY"

def test_generate_features_normalization_mock(generator, mock_client):
    # Test that the generator handles mismatched array lengths from LLM
    # The utils.normalize_array_lengths is responsible, but we test integration here
    mock_client.response_text = '{"col1": ["a"], "col2": ["b", "c"]}'
    
    # Requesting 2 rows
    df = generator.generate_features("Instr", 2, pd.DataFrame())
    
    assert len(df) == 2
    # col1 should be padded to match 2 rows
    assert df["col1"].tolist() == ["a", "a"]
    assert df["col2"].tolist() == ["b", "c"]

def test_remove_features(generator):
    df = pd.DataFrame({
        "A": [1, 2],
        "B": [3, 4],
        "C": [5, 6]
    })
    
    new_df = generator.remove_features(["B"], df)
    
    assert "B" not in new_df.columns
    assert "A" in new_df.columns
    assert "C" in new_df.columns
    assert len(new_df) == 2
    
    # Ensure original is untouched (if intended) or check return value
    # remove_features returns a COPY
    assert "B" in df.columns

def test_generate_features_invalid_json(generator, mock_client):
    mock_client.response_text = "I am not JSON"
    
    with pytest.raises(RuntimeError) as excinfo:
        generator.generate_features("Fail please", 5, pd.DataFrame())
    
    assert "Failed to parse LLM response" in str(excinfo.value)
