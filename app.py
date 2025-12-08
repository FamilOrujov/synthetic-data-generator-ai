"""
SynthGen: Streamlit UI for Synthetic Data Generation

A Streamlit-based interface for generating synthetic datasets using LLMs.
Supports: Ollama (local), OpenAI, Google Gemini, Anthropic, and Groq.
"""

import streamlit as st
import pandas as pd

from src.llm import OllamaClient, create_llm_client, get_provider_models, PROVIDER_MODELS
from src.data_generator import DataGenerator
from src.utils import export_to_csv, validate_row_count


def init_session_state():
    """Initialize session state variables."""
    if "dataframe" not in st.session_state:
        st.session_state.dataframe = pd.DataFrame()
    if "n_rows_locked" not in st.session_state:
        st.session_state.n_rows_locked = False
    if "llm_client" not in st.session_state:
        st.session_state.llm_client = None
    if "data_generator" not in st.session_state:
        st.session_state.data_generator = None
    if "template" not in st.session_state:
        st.session_state.template = ""
    if "show_index" not in st.session_state:
        st.session_state.show_index = True
    if "provider" not in st.session_state:
        st.session_state.provider = "ollama"


def setup_llm_client(
    provider: str,
    model: str,
    api_key: str = None,
    base_url: str = "http://localhost:11434",
    timeout: int = 300,
):
    """Setup the LLM client and data generator."""
    st.session_state.llm_client = create_llm_client(
        provider=provider,
        model=model,
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
    )
    st.session_state.data_generator = DataGenerator(st.session_state.llm_client)
    st.session_state.provider = provider


def main():
    st.set_page_config(
        page_title="SynthGen",
        page_icon="🧪",
        layout="wide",
    )

    st.title("Synthetic Dataset Generator AI: LLM-Powered Dataset Synthesizer")
    st.markdown("> Generate solid, realistic and well structured tabular dataset using natural language prompts.")

    init_session_state()

    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Provider selection
        st.subheader("Provider")
        provider = st.selectbox(
            "Select Provider",
            options=["ollama", "openai", "gemini", "anthropic", "groq"],
            index=0,
            format_func=lambda x: {
                "ollama": "Ollama (Local)",
                "openai": "OpenAI",
                "gemini": "Google Gemini",
                "anthropic": "Anthropic",
                "groq": "Groq",
            }.get(x, x),
            help="Choose your LLM provider.",
        )

        # Provider-specific configuration
        api_key = None
        base_url = "http://localhost:11434"
        model = None

        if provider == "ollama":
            base_url = st.text_input(
                "Ollama Base URL",
                value="http://localhost:11434",
                help="The base URL for your Ollama instance.",
            )

            # Check Ollama availability and list models
            temp_client = OllamaClient(base_url=base_url)
            if temp_client.is_available():
                st.success("✅ Ollama is running")
                available_models = temp_client.list_models()
                if available_models:
                    model = st.selectbox(
                        "Select Model",
                        options=available_models,
                        index=0,
                        help="Choose the LLM model for generation.",
                    )
                else:
                    st.warning("No models found. Please pull a model first.")
                    model = st.text_input(
                        "Model Name",
                        value="llama3.1:8b",
                        help="Enter the model name manually.",
                    )
            else:
                st.error("❌ Ollama is not running")
                st.markdown("Please start Ollama with: `ollama serve`")
                model = st.text_input(
                    "Model Name",
                    value="llama3.1:8b",
                    help="Enter the model name.",
                )

        elif provider == "openai":
            api_key = st.text_input(
                "OpenAI API Key",
                type="password",
                help="Enter your OpenAI API key.",
            )
            model = st.selectbox(
                "Select Model",
                options=get_provider_models("openai"),
                index=0,
                help="Choose the OpenAI model.",
            )
            if api_key:
                st.success("✅ API key provided")
            else:
                st.warning("⚠️ Please enter your API key")

        elif provider == "gemini":
            api_key = st.text_input(
                "Gemini API Key",
                type="password",
                help="Enter your Google Gemini API key.",
            )
            model = st.selectbox(
                "Select Model",
                options=get_provider_models("gemini"),
                index=0,
                help="Choose the Gemini model.",
            )
            if api_key:
                st.success("✅ API key provided")
            else:
                st.warning("⚠️ Please enter your API key")

        elif provider == "anthropic":
            api_key = st.text_input(
                "Anthropic API Key",
                type="password",
                help="Enter your Anthropic API key.",
            )
            model = st.selectbox(
                "Select Model",
                options=get_provider_models("anthropic"),
                index=0,
                help="Choose the Claude model.",
            )
            if api_key:
                st.success("✅ API key provided")
            else:
                st.warning("⚠️ Please enter your API key")

        elif provider == "groq":
            api_key = st.text_input(
                "Groq API Key",
                type="password",
                help="Enter your Groq API key.",
            )
            model = st.selectbox(
                "Select Model",
                options=get_provider_models("groq"),
                index=0,
                help="Choose the Groq model.",
            )
            if api_key:
                st.success("✅ API key provided")
            else:
                st.warning("⚠️ Please enter your API key")

        st.divider()

        st.subheader("Generation Settings")
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=0.7,
            step=0.1,
            help="Higher values = more creative, lower = more deterministic.",
        )
        max_tokens = st.number_input(
            "Max Tokens",
            min_value=100,
            max_value=4000,
            value=800,
            step=100,
            help="Maximum tokens in the generated response.",
        )
        timeout = st.number_input(
            "Timeout (seconds)",
            min_value=60,
            max_value=600,
            value=300,
            step=60,
            help="How long to wait for response. Increase if you get timeout errors.",
        )

        # Initialize/update client
        if st.button("Apply Settings", type="primary"):
            if provider != "ollama" and not api_key:
                st.error("Please enter your API key first!")
            else:
                try:
                    setup_llm_client(provider, model, api_key, base_url, timeout)
                    st.success(f"✅ Connected to {provider.upper()}!")
                except Exception as e:
                    st.error(f"Failed to setup client: {e}")

        # Auto-setup if not initialized (only for Ollama)
        if st.session_state.llm_client is None and provider == "ollama":
            setup_llm_client(provider, model, api_key, base_url, timeout)

        st.divider()

        # Warm-up button (only for Ollama)
        if provider == "ollama":
            st.subheader("🔥 Model Warm-up")
            st.caption("Load the model into memory before generating (recommended for first use)")
            if st.button("Warm Up Model", use_container_width=True):
                if st.session_state.llm_client:
                    with st.spinner("Loading model into memory... This may take a minute."):
                        if st.session_state.llm_client.warm_up():
                            st.success("Model ready!")
                        else:
                            st.error("Warm-up failed. Check if model is installed.")

    # Main content area
    col1, col2 = st.columns([1, 2])

    with col1:
        st.header("➕ Add Features")

        # Number of rows input
        n_rows_input = st.text_input(
            "Number of Rows",
            value="10",
            disabled=st.session_state.n_rows_locked,
            help="Number of rows to generate.",
        )

        if st.session_state.n_rows_locked:
            st.info(f"Row count locked at {len(st.session_state.dataframe)} rows.")

        # Show selected template if any
        if st.session_state.template:
            st.info(f"📋 Template: {st.session_state.template}")

        instructions = st.text_area(
            "Instructions",
            placeholder="Examples:\n• Customer data with name, email, age (25-60), and city\n• Product catalog: name, price ($10-500), category, in_stock\n• Employee records with department, salary, hire_date",
            height=150,
            help="Describe what data you want. The LLM will figure out appropriate columns and values.",
        )

        # Quick templates
        st.caption("Quick templates:")
        template_cols = st.columns(3)
        with template_cols[0]:
            if st.button("👥 People", use_container_width=True):
                st.session_state.template = "People data: first name, last name, age, email, city"
                st.rerun()
        with template_cols[1]:
            if st.button("🛒 Products", use_container_width=True):
                st.session_state.template = "Product catalog: name, price (10-500), category, stock quantity"
                st.rerun()
        with template_cols[2]:
            if st.button("🏢 Company", use_container_width=True):
                st.session_state.template = "Employee data: name, department, job title, salary, years employed"
                st.rerun()

        if st.button("🚀 Generate", type="primary", use_container_width=True):
            n_rows = validate_row_count(n_rows_input)
            final_instructions = instructions or st.session_state.get("template", "")
            if n_rows and final_instructions:
                with st.spinner("Generating data..."):
                    try:
                        new_df = st.session_state.data_generator.generate_features(
                            instructions=final_instructions,
                            n_rows=n_rows,
                            existing_dataframe=st.session_state.dataframe,
                            temperature=temperature,
                            max_tokens=max_tokens,
                        )
                        st.session_state.dataframe = new_df
                        st.session_state.n_rows_locked = True
                        st.session_state.template = ""  # Clear template after use
                        st.success("Data generated successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Generation failed: {e}")
            else:
                st.warning("Please enter valid row count and instructions (or select a template).")

        st.divider()

        # Feature management section
        st.header("🧾 Manage Data")

        if not st.session_state.dataframe.empty:
            # Index visibility toggle
            st.subheader("Display Options")
            show_index = st.checkbox(
                "Show Index Column",
                value=st.session_state.show_index,
                help="Toggle visibility of the row index column (1, 2, 3...)",
            )
            if show_index != st.session_state.show_index:
                st.session_state.show_index = show_index
                st.rerun()

            # Remove columns
            st.subheader("Remove Columns")
            columns = list(st.session_state.dataframe.columns)
            selected_features = st.multiselect(
                "Select columns to remove",
                options=columns,
                help="Select columns to remove from the dataset.",
            )

            if st.button("🗑️ Remove Columns", use_container_width=True):
                if selected_features:
                    st.session_state.dataframe = st.session_state.data_generator.remove_features(
                        selected_features, st.session_state.dataframe
                    )
                    if st.session_state.dataframe.empty:
                        st.session_state.n_rows_locked = False
                    st.success("Columns removed!")
                    st.rerun()
                else:
                    st.warning("No columns selected.")

            # Remove rows
            st.subheader("Remove Rows")
            row_count = len(st.session_state.dataframe)
            row_indices = list(range(1, row_count + 1))  # 1-based indices for display
            selected_rows = st.multiselect(
                "Select rows to remove",
                options=row_indices,
                help="Select row numbers to remove from the dataset.",
            )

            if st.button("🗑️ Remove Rows", use_container_width=True):
                if selected_rows:
                    # Convert 1-based to 0-based positional indices
                    indices_to_keep = [i for i in range(row_count) if (i + 1) not in selected_rows]
                    st.session_state.dataframe = st.session_state.dataframe.iloc[indices_to_keep].reset_index(drop=True)
                    if st.session_state.dataframe.empty:
                        st.session_state.n_rows_locked = False
                    st.success(f"Removed {len(selected_rows)} row(s)!")
                    st.rerun()
                else:
                    st.warning("No rows selected.")
        else:
            st.info("No data generated yet.")

        st.divider()

        # Export section
        st.header("💾 Export")

        if not st.session_state.dataframe.empty:
            if st.button("📥 Export to CSV", use_container_width=True):
                filename = export_to_csv(st.session_state.dataframe)
                st.success(f"Exported to {filename}")

            # Also provide download button
            csv_data = st.session_state.dataframe.to_csv(index=False)
            st.download_button(
                label="⬇️ Download CSV",
                data=csv_data,
                file_name="synthetic_dataset.csv",
                mime="text/csv",
                use_container_width=True,
            )
        else:
            st.info("Generate data first to export.")

        st.divider()

        # Reset button
        if st.button("🔄 Reset All", use_container_width=True):
            st.session_state.dataframe = pd.DataFrame()
            st.session_state.n_rows_locked = False
            st.rerun()

    with col2:
        st.header("📊 Generated Dataset")

        if not st.session_state.dataframe.empty:
            # Display dataframe with optional index
            display_df = st.session_state.dataframe.copy()
            if st.session_state.show_index:
                display_df.index = range(1, len(display_df) + 1)
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    height=500,
                )
            else:
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    height=500,
                    hide_index=True,
                )

            # Dataset info
            st.markdown(f"**Shape:** {st.session_state.dataframe.shape[0]} rows × {st.session_state.dataframe.shape[1]} columns")
            st.markdown(f"**Columns:** {', '.join(st.session_state.dataframe.columns)}")
        else:
            st.info("No data generated yet. Use the panel on the left to generate synthetic data.")

            # Show example usage
            with st.expander("📖 How to use"):
                st.markdown("""
                1. **Configure Ollama** in the sidebar (ensure Ollama is running)
                2. **Enter the number of rows** you want to generate
                3. **Describe your data** in natural language:
                   - Column names
                   - Value types and ranges
                   - Constraints and relationships
                4. **Click Generate** to create your dataset
                5. **Add more features** by generating again with new instructions
                6. **Remove unwanted columns** using the Manage Features section
                7. **Export** your dataset to CSV

                ### Example Instructions:
                ```
                Generate 3 columns: first_name, last_name, age.
                Male names should be between 30-50 years old.
                Female names should be between 25-40 years old.
                Make sure gender and age correlate row-wise.
                ```
                """)


if __name__ == "__main__":
    main()

