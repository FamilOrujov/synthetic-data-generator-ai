"""
Utility Functions Module

Provides helper functions for JSON extraction and data export.
"""

import json
import os
import re
from typing import Optional

import pandas as pd


def extract_json_response(text: str) -> Optional[dict]:
    """
    Extract a valid JSON object from model output text.

    This function searches for the last valid JSON object in the text,
    which is useful if the model generates additional text or explanations
    that are not part of the JSON response.

    Args:
        text: The raw text output from the model.

    Returns:
        A dictionary parsed from the JSON, or None if no valid JSON found.
    """
    decoder = json.JSONDecoder()
    starts = [m.start() for m in re.finditer(r"{", text)]

    for pos in reversed(starts):  # Start from last candidate
        try:
            parsed, _ = decoder.raw_decode(text[pos:])
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue

    return None


def export_to_csv(dataframe: pd.DataFrame, base_filename: str = "dataset") -> str:
    """
    Export a DataFrame to a CSV file with auto-incrementing filename.

    Args:
        dataframe: The pandas DataFrame to export.
        base_filename: The base name for the output file.

    Returns:
        The filename that was written to.
    """
    n = 0
    while True:
        if n == 0:
            filename = f"{base_filename}.csv"
        else:
            filename = f"{base_filename}{n}.csv"

        if not os.path.exists(filename):
            break
        n += 1

    dataframe.to_csv(filename, index=False)
    return filename


def validate_row_count(n_rows: str) -> Optional[int]:
    """
    Validate and parse the row count input.

    Args:
        n_rows: The row count as a string.

    Returns:
        The parsed integer if valid, None otherwise.
    """
    if n_rows.isnumeric() and int(n_rows) >= 1:
        return int(n_rows)
    return None


def normalize_array_lengths(data: dict, target_length: int) -> dict:
    """
    Normalize all arrays in a dictionary to have the same target length.

    If an array is shorter, it will be padded by repeating values.
    If an array is longer, it will be truncated.

    Args:
        data: Dictionary with column names as keys and lists as values.
        target_length: The desired length for all arrays.

    Returns:
        Dictionary with all arrays normalized to target_length.
    """
    normalized = {}

    for key, values in data.items():
        if not isinstance(values, list):
            # Convert single value to list
            values = [values]

        current_len = len(values)

        if current_len == 0:
            # Fill with None if empty
            normalized[key] = [None] * target_length
        elif current_len == target_length:
            normalized[key] = values
        elif current_len < target_length:
            # Pad by cycling through existing values
            cycles = (target_length // current_len) + 1
            extended = (values * cycles)[:target_length]
            normalized[key] = extended
        else:
            # Truncate if too long
            normalized[key] = values[:target_length]

    return normalized

