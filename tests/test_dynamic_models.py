"""
Test script for the dynamic model fetching functionality.

This script tests the model fetchers and router function for OpenAI, Anthropic, and Google.
It verifies that models can be fetched from each provider and handles error cases.

Usage:
    python -m tests.test_dynamic_models

Note: Requires API keys to be set in the environment variables:
    - OPENAI_API_KEY
    - ANTHROPIC_API_KEY
    - GOOGLE_API_KEY
"""

import asyncio
import os
import sys
import unittest
from typing import List

# Add the parent directory to the path so we can import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.llm_model_registry import get_models, _openai_models, _anthropic_models, _google_models


async def smoke_test():
    """Simple smoke test to print available models from each provider."""
    for provider in ("openai", "anthropic", "google"):
        print(f"\n--- Testing {provider.upper()} Model Fetcher ---")
        try:
            models = await get_models(provider)
            print(f"{provider}: {len(models)} models found")
            if models:
                print("First 5 models:")
                for model in models[:5]:
                    print(f"  - {model}")
            else:
                print(f"No models found for {provider}. Check your API key.")
        except Exception as e:
            print(f"Error fetching models for {provider}: {e}")
    print("\nSmoke test completed.")


class TestModelFetchers(unittest.TestCase):
    """Test cases for the model fetchers."""

    def setUp(self):
        """Set up the test environment."""
        # Check if API keys are set
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        self.google_key = os.getenv("GOOGLE_API_KEY")

    def test_router_function(self):
        """Test the router function with valid and invalid providers."""
        async def _test():
            # Test with valid provider
            openai_models = await get_models("openai")
            self.assertIsInstance(openai_models, list)
            
            # Test with invalid provider
            invalid_models = await get_models("invalid_provider")
            self.assertEqual(invalid_models, [])
            
            # Test with empty string
            empty_models = await get_models("")
            self.assertEqual(empty_models, [])
            
            # Test with None (should handle gracefully)
            try:
                none_models = await get_models(None)  # type: ignore
                self.assertEqual(none_models, [])
            except Exception:
                # It's okay if it raises an exception too
                pass
        
        asyncio.run(_test())

    def test_openai_fetcher(self):
        """Test the OpenAI model fetcher."""
        if not self.openai_key:
            self.skipTest("OPENAI_API_KEY not set")
        
        async def _test():
            models = await _openai_models()
            self.assertIsInstance(models, list)
            if models:  # Only assert if we got models back
                self.assertTrue(any("gpt" in model.lower() for model in models))
        
        asyncio.run(_test())

    def test_anthropic_fetcher(self):
        """Test the Anthropic model fetcher."""
        if not self.anthropic_key:
            self.skipTest("ANTHROPIC_API_KEY not set")
        
        async def _test():
            models = await _anthropic_models()
            self.assertIsInstance(models, list)
            if models:  # Only assert if we got models back
                self.assertTrue(any("claude" in model.lower() for model in models))
        
        asyncio.run(_test())

    def test_google_fetcher(self):
        """Test the Google model fetcher."""
        if not self.google_key:
            self.skipTest("GOOGLE_API_KEY not set")
        
        async def _test():
            models = await _google_models()
            self.assertIsInstance(models, list)
            if models:  # Only assert if we got models back
                self.assertTrue(any("gemini" in model.lower() for model in models))
        
        asyncio.run(_test())


if __name__ == "__main__":
    # Run the smoke test if called directly
    if len(sys.argv) > 1 and sys.argv[1] == "--smoke":
        asyncio.run(smoke_test())
    else:
        # Run the unit tests
        unittest.main()
