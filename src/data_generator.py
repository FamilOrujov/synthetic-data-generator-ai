"""
Data Generator Module

Core logic for synthetic data generation using LLMs.
"""

import pandas as pd

from .llm import OllamaClient
from .utils import extract_json_response, normalize_array_lengths


# System prompt for the LLM to generate structured data
SYSTEM_PROMPT = """You are a synthetic dataset generator. Generate realistic tabular data based on user descriptions.

CRITICAL RULES:
1. Output ONLY valid JSON - no explanations, no markdown code blocks, no extra text before or after
2. Start your response with { and end with }
3. EVERY column array MUST have EXACTLY the same number of elements (matching the requested row count)
4. Column names should be lowercase with underscores (snake_case)
5. Do NOT wrap JSON in ```json or ``` markers

OUTPUT FORMAT:
{
  "column_name": ["value1", "value2", ...],
  "another_column": [1, 2, ...]
}

GUIDELINES:
- If the user asks for "names", generate both first_name and last_name columns
- If the user mentions "age", use realistic age ranges (18-80 for adults)
- If the user asks for "email", generate realistic email addresses
- Make data realistic and internally consistent (e.g., names match gender if specified)
- For numeric ranges, distribute values naturally across the range
- If constraints are unclear, make reasonable assumptions

EXAMPLE - User asks for "5 rows of people data":
{
  "first_name": ["Alice", "Bob", "Carol", "David", "Emma"],
  "last_name": ["Smith", "Johnson", "Williams", "Brown", "Jones"],
  "age": [28, 35, 42, 31, 27],
  "email": ["alice.smith@email.com", "bob.j@mail.com", "carol.w@inbox.com", "david.b@email.com", "emma.jones@mail.com"]
}

Remember: Output ONLY the JSON object. EXACTLY N rows per column where N is the requested count."""


class DataGenerator:
    """Generates synthetic datasets using an LLM backend."""

    def __init__(self, llm_client: OllamaClient):
        """
        Initialize the DataGenerator.

        Args:
            llm_client: An OllamaClient instance for LLM interactions.
        """
        self.llm_client = llm_client
        self.system_prompt = SYSTEM_PROMPT

    def build_user_query(self, instructions: str, n_rows: int) -> str:
        """
        Build the user query for the LLM.

        Args:
            instructions: User-provided instructions for data generation.
            n_rows: Number of rows to generate.

        Returns:
            The formatted user query string.
        """
        return f"Generate EXACTLY {n_rows} rows of data.\n\nRequirements: {instructions.strip()}"

    def generate_features(
        self,
        instructions: str,
        n_rows: int,
        existing_dataframe: pd.DataFrame,
        temperature: float = 0.7,
        max_tokens: int = 800,
    ) -> pd.DataFrame:
        """
        Generate new features based on user instructions.

        Args:
            instructions: Natural language instructions for data generation.
            n_rows: Number of rows to generate.
            existing_dataframe: Existing DataFrame to append features to.
            temperature: Sampling temperature for generation.
            max_tokens: Maximum tokens for generation.

        Returns:
            Updated DataFrame with new features added.

        Raises:
            ValueError: If row count mismatch between existing and new data.
            RuntimeError: If LLM fails to generate valid JSON response.
        """
        # Use existing dataframe length if locked, otherwise use requested n_rows
        target_rows = len(existing_dataframe) if len(existing_dataframe) > 0 else n_rows

        user_query = self.build_user_query(instructions, target_rows)

        # Generate response from LLM
        response = self.llm_client.generate(
            prompt=user_query,
            system_prompt=self.system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Parse the response
        parsed_data = extract_json_response(response)
        if parsed_data is None:
            error_preview = response[:200] if len(response) > 200 else response
            raise RuntimeError(
                f"Failed to parse LLM response as valid JSON. "
                f"Response preview: {error_preview}... "
                f"Try increasing max_tokens or using a different model."
            )

        # Normalize array lengths to ensure all columns have same length
        normalized_data = normalize_array_lengths(parsed_data, target_rows)

        new_df = pd.DataFrame(normalized_data)

        # Combine with existing data
        if len(existing_dataframe) == 0:
            return new_df
        else:
            return pd.concat(
                [existing_dataframe.reset_index(drop=True), new_df.reset_index(drop=True)],
                axis=1,
            )

    def remove_features(
        self, selected_features: list[str], existing_dataframe: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Remove selected features from the DataFrame.

        Args:
            selected_features: List of column names to remove.
            existing_dataframe: The DataFrame to modify.

        Returns:
            DataFrame with selected columns removed.
        """
        if not selected_features:
            return existing_dataframe.copy()

        edited_df = existing_dataframe.copy()
        edited_df.drop(columns=selected_features, axis=1, inplace=True)

        return edited_df

