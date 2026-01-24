"""
LLM Client Module

Provides interfaces to interact with various LLM providers:
- Ollama (local)
- OpenAI
- Google Gemini
- Anthropic
- Groq
"""

import time
import requests
from typing import Optional


# Available models for each provider
PROVIDER_MODELS = {
    "ollama": [],  # Dynamically loaded
    "openai": [
        "gpt-4.1",
        "gpt-4.1-mini",
        "gpt-4.1-nano",
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4.5",
        "gpt-5",
        "gpt-5.1",
        "o1",
        "o1-mini",
        "o3",
        "o3-mini",
        "o4-mini",
    ],
    "gemini": [
        "gemini-2.0-flash",
        "gemini-1.5-pro",
        "gemini-1.5-flash",
        "gemini-2.5-flash-lite-preview-09-2025",
        "gemini-3-pro-preview",
    ],
    "anthropic": [
        "claude-sonnet-4-5",
        "claude-haiku-4-5",
        "claude-opus-4-5",
        "claude-sonnet-4-5-20250929",
        "claude-haiku-4-5-20251001",
        "claude-opus-4-5-20251101",
        "claude-3-haiku-20240307",
        "claude-3-5-haiku-20241022",
    ],
    "groq": [
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile",
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
        "meta-llama/llama-guard-4-12b",
        "openai/gpt-oss-safeguard-20b",
        "meta-llama/llama-4-maverick-17b-128e-instruct",
        "meta-llama/llama-4-scout-17b-16e-instruct",
        "qwen/qwen3-32b",
        "moonshotai/kimi-k2-instruct-0905",
    ],
}


def get_provider_models(provider: str) -> list[str]:
    """Get available models for a provider."""
    return PROVIDER_MODELS.get(provider, [])


class OllamaClient:
    """Client for interacting with Ollama API for local LLM inference."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.1:8b",
        timeout: int = 300,
        max_retries: int = 3,
    ):
        """
        Initialize the Ollama client.

        Args:
            base_url: The base URL for the Ollama API.
            model: The model name to use for generation.
            timeout: Request timeout in seconds (default 300 = 5 minutes).
            max_retries: Maximum number of retry attempts on failure.
        """
        self.base_url = base_url
        self.model = model
        self.api_endpoint = f"{base_url}/api/generate"
        self.timeout = timeout
        self.max_retries = max_retries

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 800,
    ) -> str:
        """
        Generate a response from the LLM with retry logic.

        Args:
            prompt: The user prompt to send to the model.
            system_prompt: Optional system prompt to guide the model's behavior.
            temperature: Sampling temperature for generation.
            max_tokens: Maximum number of tokens to generate.

        Returns:
            The generated text response from the model.
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        if system_prompt:
            payload["system"] = system_prompt

        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.api_endpoint,
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                result = response.json()
                return result.get("response", "")
            except requests.exceptions.Timeout:
                last_error = f"Request timed out after {self.timeout}s (attempt {attempt + 1}/{self.max_retries})"
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff: 1s, 2s, 4s
            except requests.exceptions.ConnectionError as e:
                last_error = f"Connection error: {e}"
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
            except requests.exceptions.RequestException as e:
                last_error = str(e)
                break  # Don't retry on other errors

        raise ConnectionError(f"Failed to connect to Ollama: {last_error}")

    def warm_up(self) -> bool:
        """
        Warm up the model by sending a minimal request.

        This helps load the model into memory before the first real request.

        Returns:
            True if warm-up succeeded, False otherwise.
        """
        try:
            payload = {
                "model": self.model,
                "prompt": "Hi",
                "stream": False,
                "options": {"num_predict": 1},
            }
            response = requests.post(self.api_endpoint, json=payload, timeout=180)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def list_models(self) -> list[str]:
        """
        List available models in Ollama.

        Returns:
            A list of available model names.
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            response.raise_for_status()
            result = response.json()
            models = result.get("models", [])
            return [model.get("name", "") for model in models]
        except requests.exceptions.RequestException:
            return []

    def is_available(self) -> bool:
        """
        Check if Ollama is available and running.

        Returns:
            True if Ollama is available, False otherwise.
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False


class OpenAIClient:
    """Client for interacting with OpenAI API."""

    def __init__(self, api_key: str, model: str = "gpt-4o", timeout: int = 120):
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.api_endpoint = "https://api.openai.com/v1/chat/completions"

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 800,
    ) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        try:
            response = requests.post(
                self.api_endpoint, headers=headers, json=payload, timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"OpenAI API error: {e}")

    def warm_up(self) -> bool:
        return True  # No warm-up needed for cloud APIs


class GeminiClient:
    """Client for interacting with Google Gemini API."""

    def __init__(self, api_key: str, model: str = "gemini-2.0-flash", timeout: int = 120):
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 800,
    ) -> str:
        api_endpoint = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        )

        # Combine system prompt and user prompt
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt

        payload = {
            "contents": [{"parts": [{"text": full_prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }

        try:
            response = requests.post(api_endpoint, json=payload, timeout=self.timeout)
            response.raise_for_status()
            result = response.json()
            return result["candidates"][0]["content"]["parts"][0]["text"]
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Gemini API error: {e}")

    def warm_up(self) -> bool:
        return True


class AnthropicClient:
    """Client for interacting with Anthropic API."""

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-5", timeout: int = 120):
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.api_endpoint = "https://api.anthropic.com/v1/messages"

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 800,
    ) -> str:
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }

        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }

        if system_prompt:
            payload["system"] = system_prompt

        # Note: Some Claude models don't support temperature
        if "o1" not in self.model and "o3" not in self.model:
            payload["temperature"] = temperature

        try:
            response = requests.post(
                self.api_endpoint, headers=headers, json=payload, timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()
            return result["content"][0]["text"]
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Anthropic API error: {e}")

    def warm_up(self) -> bool:
        return True


class GroqClient:
    """Client for interacting with Groq API."""

    def __init__(self, api_key: str, model: str = "llama-3.1-8b-instant", timeout: int = 120):
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.api_endpoint = "https://api.groq.com/openai/v1/chat/completions"

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 800,
    ) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        try:
            response = requests.post(
                self.api_endpoint, headers=headers, json=payload, timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Groq API error: {e}")

    def warm_up(self) -> bool:
        return True


def create_llm_client(
    provider: str,
    model: str,
    api_key: Optional[str] = None,
    base_url: str = "http://localhost:11434",
    timeout: int = 300,
):
    """
    Factory function to create the appropriate LLM client.

    Args:
        provider: The provider name (ollama, openai, gemini, anthropic, groq).
        model: The model name to use.
        api_key: API key for cloud providers.
        base_url: Base URL for Ollama.
        timeout: Request timeout in seconds.

    Returns:
        An LLM client instance.
    """
    if provider == "ollama":
        return OllamaClient(base_url=base_url, model=model, timeout=timeout)
    elif provider == "openai":
        if not api_key:
            raise ValueError("OpenAI API key is required")
        return OpenAIClient(api_key=api_key, model=model, timeout=timeout)
    elif provider == "gemini":
        if not api_key:
            raise ValueError("Gemini API key is required")
        return GeminiClient(api_key=api_key, model=model, timeout=timeout)
    elif provider == "anthropic":
        if not api_key:
            raise ValueError("Anthropic API key is required")
        return AnthropicClient(api_key=api_key, model=model, timeout=timeout)
    elif provider == "groq":
        if not api_key:
            raise ValueError("Groq API key is required")
        return GroqClient(api_key=api_key, model=model, timeout=timeout)
    else:
        raise ValueError(f"Unknown provider: {provider}")
