import pytest
from src.utils import validate_row_count, normalize_array_lengths, extract_json_response

def test_validate_row_count():
    assert validate_row_count("10") == 10
    assert validate_row_count("1") == 1
    assert validate_row_count("0") is None
    assert validate_row_count("-5") is None
    assert validate_row_count("abc") is None
    assert validate_row_count("10.5") is None

def test_normalize_array_lengths():
    # Case 1: All lengths match
    data = {"col1": [1, 2], "col2": ["a", "b"]}
    normalized = normalize_array_lengths(data, 2)
    assert normalized == data

    # Case 2: One column shorter (padding)
    data_short = {"col1": [1], "col2": ["a", "b"]}
    normalized_short = normalize_array_lengths(data_short, 2)
    assert normalized_short["col1"] == [1, 1]

    # Case 3: One column longer (truncation)
    data_long = {"col1": [1, 2, 3], "col2": ["a", "b"]}
    normalized_long = normalize_array_lengths(data_long, 2)
    assert normalized_long["col1"] == [1, 2]

    # Case 4: Single value not in list
    data_single = {"col1": 1, "col2": ["a", "b"]}
    normalized_single = normalize_array_lengths(data_single, 2)
    assert normalized_single["col1"] == [1, 1]
    
    # Case 5: Empty input
    data_empty = {"col1": []}
    normalized_empty = normalize_array_lengths(data_empty, 3)
    assert normalized_empty["col1"] == [None, None, None]

def test_extract_json_response():
    # Valid JSON
    assert extract_json_response('{"a": 1}') == {"a": 1}
    
    # JSON in markdown block
    assert extract_json_response('```json\n{"a": 1}\n```') == {"a": 1}
    
    # Text around JSON
    assert extract_json_response('Here is the data: {"a": 1} thanks') == {"a": 1}
    
    # Invalid JSON
    assert extract_json_response('d{a: 1}') is None
    assert extract_json_response('') is None
