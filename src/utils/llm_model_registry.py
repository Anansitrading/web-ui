"""
LLM Model Registry - Dynamic Model Fetching Module

This module provides functions to dynamically fetch available models from various LLM providers:
- OpenAI
- Anthropic
- Google (Gemini/Vertex-AI)

The module implements caching with TTL to avoid excessive API calls and provides a unified
interface for fetching models from any supported provider.
"""

from __future__ import annotations
import os
import asyncio
import logging
import httpx
from async_lru import alru_cache
from typing import List
from datetime import timedelta

# TTL in seconds – adjust per need
CACHE_TTL = 3600  # 1 hour cache

# ───────────────────────────────── OpenAI ──────────────────────────────────
@alru_cache(ttl=CACHE_TTL)
async def _openai_models() -> List[str]:
    """
    Fetch available models from OpenAI API.
    
    Returns:
        List[str]: Sorted list of model IDs
    """
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        logging.warning("OpenAI API key is not set. Cannot fetch models.")
        return []
        
    try:
        import openai
        client = openai.AsyncOpenAI(
            api_key=api_key,
            # endpoint override allowed via env
            base_url=os.getenv("OPENAI_ENDPOINT", "https://api.openai.com/v1"),
            timeout=httpx.Timeout(10.0, connect=5.0)  # Set timeout to avoid hanging
        )
        # handles automatic pagination
        objs = await client.models.list()
        return sorted(m.id for m in objs.data)
    except httpx.TimeoutException:
        logging.error("Timeout while fetching OpenAI models. Network may be slow or API unresponsive.")
        return []
    except openai.AuthenticationError as e:
        logging.error(f"Authentication error with OpenAI API: {e}")
        return []
    except openai.RateLimitError as e:
        logging.error(f"Rate limit exceeded for OpenAI API: {e}")
        return []
    except Exception as e:
        logging.error(f"Error fetching OpenAI models: {e}")
        return []

# ───────────────────────────────── Anthropic ───────────────────────────────
@alru_cache(ttl=CACHE_TTL)
async def _anthropic_models() -> List[str]:
    """
    Fetch available models from Anthropic API.
    
    Returns:
        List[str]: Sorted list of model IDs
    """
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        logging.warning("Anthropic API key is not set. Cannot fetch models.")
        return []
        
    try:
        import anthropic
        client = anthropic.AsyncAnthropic(
            api_key=api_key,
            base_url=os.getenv("ANTHROPIC_ENDPOINT", "https://api.anthropic.com"),
            timeout=httpx.Timeout(10.0, connect=5.0)  # Set timeout to avoid hanging
        )
        # AsyncPage[ModelInfo] object needs to be iterated
        resp = await client.models.list()
        models = []
        async for model in resp:
            models.append(model.id)
        return sorted(models)
    except httpx.TimeoutException:
        logging.error("Timeout while fetching Anthropic models. Network may be slow or API unresponsive.")
        return []
    except anthropic.AuthenticationError as e:
        logging.error(f"Authentication error with Anthropic API: {e}")
        return []
    except anthropic.RateLimitError as e:
        logging.error(f"Rate limit exceeded for Anthropic API: {e}")
        return []
    except Exception as e:
        logging.error(f"Error fetching Anthropic models: {e}")
        return []

# ───────────────────────────────── Google (Gemini AI Studio) ───────────────
@alru_cache(ttl=CACHE_TTL)
async def _google_models() -> List[str]:
    """
    Fetch available models from Google Gemini API.
    
    Returns:
        List[str]: Sorted list of model names
    """
    api_key = os.getenv("GOOGLE_API_KEY", "")
    if not api_key:
        logging.warning("Google API key is not set. Cannot fetch models.")
        return []
    
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        
        # Set a timeout for the executor task
        async def fetch_with_timeout():
            # This call is synchronous; wrap in thread so our API stays async
            loop = asyncio.get_running_loop()
            try:
                models = await asyncio.wait_for(
                    loop.run_in_executor(None, genai.list_models),
                    timeout=10.0  # 10 second timeout
                )
                # Each model has a `name` like 'gemini-1.5-flash-latest'
                return sorted(m.name for m in models)
            except asyncio.TimeoutError:
                logging.error("Timeout while fetching Google models. Network may be slow or API unresponsive.")
                return []
        
        return await fetch_with_timeout()
    except ImportError as e:
        logging.error(f"Google Generative AI package not installed: {e}")
        return []
    except ValueError as e:
        logging.error(f"Invalid API key or configuration for Google API: {e}")
        return []
    except Exception as e:
        logging.error(f"Error fetching Google models: {e}")
        return []

# ───────────────────────────────── Mistral ──────────────────────────────────
@alru_cache(ttl=CACHE_TTL)
async def _mistral_models() -> List[str]:
    """
    Fetch available models from Mistral API.
    
    Returns:
        List[str]: Sorted list of model IDs
    """
    api_key = os.getenv("MISTRAL_API_KEY", "")
    if not api_key:
        logging.warning("Mistral API key is not set. Cannot fetch models.")
        return []
        
    try:
        import httpx
        async with httpx.AsyncClient(timeout=10.0) as client:
            client.headers.update({
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            })
            endpoint = os.getenv("MISTRAL_ENDPOINT", "https://api.mistral.ai/v1")
            response = await client.get(f"{endpoint}/models")
            response.raise_for_status()
            data = response.json()
            return sorted(model["id"] for model in data.get("data", []))
    except httpx.TimeoutException:
        logging.error("Timeout while fetching Mistral models. Network may be slow or API unresponsive.")
        return []
    except httpx.HTTPStatusError as e:
        logging.error(f"HTTP error while fetching Mistral models: {e}")
        return []
    except Exception as e:
        logging.error(f"Error fetching Mistral models: {e}")
        return []

# ───────────────────────────────── DeepSeek ──────────────────────────────────
@alru_cache(ttl=CACHE_TTL)
async def _deepseek_models() -> List[str]:
    """
    Fetch available models from DeepSeek API.
    
    Returns:
        List[str]: Sorted list of model IDs
    """
    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    if not api_key:
        logging.warning("DeepSeek API key is not set. Cannot fetch models.")
        return []
        
    try:
        import httpx
        async with httpx.AsyncClient(timeout=10.0) as client:
            client.headers.update({
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            })
            endpoint = os.getenv("DEEPSEEK_ENDPOINT", "https://api.deepseek.com")
            response = await client.get(f"{endpoint}/compatible-mode/v1/models")
            response.raise_for_status()
            data = response.json()
            return sorted(model["id"] for model in data.get("data", []))
    except httpx.TimeoutException:
        logging.error("Timeout while fetching DeepSeek models. Network may be slow or API unresponsive.")
        return []
    except httpx.HTTPStatusError as e:
        logging.error(f"HTTP error while fetching DeepSeek models: {e}")
        return []
    except Exception as e:
        logging.error(f"Error fetching DeepSeek models: {e}")
        return []

# ───────────────────────────────── Azure OpenAI ──────────────────────────────────
@alru_cache(ttl=CACHE_TTL)
async def _azure_openai_models() -> List[str]:
    """
    Fetch available models from Azure OpenAI API.
    
    Returns:
        List[str]: Sorted list of model IDs
    """
    api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    if not api_key or not endpoint:
        logging.warning("Azure OpenAI API key or endpoint is not set. Cannot fetch models.")
        return []
        
    try:
        import openai
        client = openai.AsyncAzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview"),
            timeout=httpx.Timeout(10.0, connect=5.0)  # Set timeout to avoid hanging
        )
        # handles automatic pagination
        objs = await client.models.list()
        return sorted(m.id for m in objs.data)
    except httpx.TimeoutException:
        logging.error("Timeout while fetching Azure OpenAI models. Network may be slow or API unresponsive.")
        return []
    except openai.AuthenticationError as e:
        logging.error(f"Authentication error with Azure OpenAI API: {e}")
        return []
    except openai.RateLimitError as e:
        logging.error(f"Rate limit exceeded for Azure OpenAI API: {e}")
        return []
    except Exception as e:
        logging.error(f"Error fetching Azure OpenAI models: {e}")
        return []

# ───────────────────────────────── Router ──────────────────────────────────
async def get_models(provider: str) -> List[str]:
    """
    Get available models for the specified provider.
    
    Args:
        provider (str): Provider ID ('openai', 'anthropic', 'google', 'mistral', 'deepseek', 'azure_openai')
        or any custom provider registered with register_model_fetcher()
        
    Returns:
        List[str]: List of available model IDs/names
    """
    # Validate input
    if not provider or not isinstance(provider, str):
        logging.warning(f"Invalid provider specified: {provider}")
        return []
    
    provider = provider.lower()
    
    # Map provider to fetcher function
    built_in_mapping = {
        "openai":       _openai_models,
        "anthropic":    _anthropic_models,
        "google":       _google_models,
        "mistral":      _mistral_models,
        "deepseek":     _deepseek_models,
        "azure_openai": _azure_openai_models,
    }
    
    # Check custom fetchers first, then fall back to built-in fetchers
    fn = _custom_fetchers.get(provider) or built_in_mapping.get(provider)
    
    if not fn:
        logging.warning(f"No model fetcher for provider: {provider}")
        return []
    
    try:
        # Fetch models with a timeout to prevent hanging
        models = await asyncio.wait_for(fn(), timeout=15.0)  # 15 second overall timeout
        
        # Handle empty model lists (possible API issue)
        if not models:
            logging.warning(f"No models returned for {provider}. API may be having issues or key may not have access.")
            
            # Try to get stale cache if available
            try:
                cached_models = await fn.cache_get(provider)  # type: ignore[attr-defined]
                if cached_models:
                    logging.info(f"Using cached models for {provider} from previous successful fetch.")
                    return cached_models
            except Exception:
                pass
        
        return models
    except asyncio.TimeoutError:
        logging.error(f"Timeout while fetching {provider} models. Operation took too long.")
        # Try to get stale cache if available
        try:
            cached_models = await fn.cache_get(provider)  # type: ignore[attr-defined]
            if cached_models:
                logging.info(f"Using cached models for {provider} from previous successful fetch.")
                return cached_models
        except Exception:
            pass
        return []
    except Exception as e:
        logging.error(f"Fetching {provider} models failed: {e}", exc_info=True)
        # Try to get stale cache if available
        try:
            cached_models = await fn.cache_get(provider)  # type: ignore[attr-defined]
            if cached_models:
                logging.info(f"Using cached models for {provider} from previous successful fetch.")
                return cached_models
        except Exception:
            pass
        return []

# Registry for custom model fetchers
_custom_fetchers = {}

def register_model_fetcher(provider_id: str, fetcher_func):
    """
    Register a custom model fetcher function for a provider.
    
    This allows for extending the model fetching system with custom providers
    without modifying the core code.
    
    Args:
        provider_id (str): The provider ID to register the fetcher for
        fetcher_func (callable): An async function that returns a list of model IDs/names
        
    Example:
        ```python
        @alru_cache(ttl=3600)
        async def _my_custom_provider_models() -> List[str]:
            # Custom implementation
            return ["model-1", "model-2"]
            
        register_model_fetcher("my_custom_provider", _my_custom_provider_models)
        ```
    """
    global _custom_fetchers
    _custom_fetchers[provider_id.lower()] = fetcher_func
    logging.info(f"Registered custom model fetcher for provider: {provider_id}")


# Function to configure Redis caching instead of in-memory caching
def configure_redis_cache(redis_url: str = "redis://localhost:6379/0", ttl: int = 3600):
    """
    Configure Redis as a cache backend for model fetching instead of in-memory caching.
    
    This is useful for distributed environments where multiple instances of the application
    can share the same cache.
    
    Args:
        redis_url (str): Redis connection URL
        ttl (int): Time-to-live for cache entries in seconds
        
    Note:
        This function requires the redis package to be installed.
        You can install it with: pip install redis
    """
    try:
        import redis
        from functools import wraps
        import pickle
        import json
        
        # Create Redis client
        redis_client = redis.from_url(redis_url)
        
        # Test connection
        redis_client.ping()
        
        # Create a decorator that uses Redis for caching
        def redis_cache(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Create a cache key based on function name and arguments
                key = f"model_registry:{func.__name__}:{str(args)}:{str(kwargs)}"
                
                # Try to get from cache
                cached = redis_client.get(key)
                if cached:
                    try:
                        return pickle.loads(cached)
                    except:
                        # Fallback to JSON if pickle fails
                        return json.loads(cached)
                
                # Call the function
                result = await func(*args, **kwargs)
                
                # Store in cache
                try:
                    redis_client.setex(key, ttl, pickle.dumps(result))
                except:
                    # Fallback to JSON if pickle fails
                    redis_client.setex(key, ttl, json.dumps(result))
                
                return result
            
            # Add a method to get cached value
            async def cache_get(*args, **kwargs):
                key = f"model_registry:{func.__name__}:{str(args)}:{str(kwargs)}"
                cached = redis_client.get(key)
                if cached:
                    try:
                        return pickle.loads(cached)
                    except:
                        return json.loads(cached)
                raise KeyError(f"No cached value for {key}")
            
            wrapper.cache_get = cache_get
            return wrapper
        
        # Replace the alru_cache decorator with our Redis cache
        global _openai_models, _anthropic_models, _google_models, _mistral_models, _deepseek_models, _azure_openai_models
        
        # Store the original functions
        original_functions = {
            "openai": _openai_models,
            "anthropic": _anthropic_models,
            "google": _google_models,
            "mistral": _mistral_models,
            "deepseek": _deepseek_models,
            "azure_openai": _azure_openai_models,
        }
        
        # Replace with Redis-cached versions
        for name, func in original_functions.items():
            # Get the original function (unwrapped)
            original = func.__wrapped__
            # Create a new Redis-cached version
            redis_cached = redis_cache(original)
            # Register it in the custom fetchers
            register_model_fetcher(name, redis_cached)
        
        logging.info(f"Configured Redis cache for model fetching at {redis_url} with TTL {ttl}s")
        return True
    except ImportError:
        logging.error("Redis package not installed. Install with: pip install redis")
        return False
    except Exception as e:
        logging.error(f"Failed to configure Redis cache: {e}")
        return False


# Optional: Filter models by capability (e.g., chat-capable models)
def _filter_chat(models: List[str], provider: str) -> List[str]:
    """
    Filter models to only include chat-capable models.
    
    Args:
        models (List[str]): List of model IDs/names
        provider (str): Provider ID
        
    Returns:
        List[str]: Filtered list of chat-capable models
    """
    if provider == "openai":
        return [m for m in models if "gpt" in m]
    if provider == "anthropic":
        return [m for m in models if m.startswith("claude")]
    if provider == "google":
        return [m for m in models if "gemini" in m]
    if provider == "mistral":
        return [m for m in models if "mistral" in m.lower()]
    if provider == "deepseek":
        return [m for m in models if "deepseek" in m.lower()]
    if provider == "azure_openai":
        return [m for m in models if "gpt" in m]
    return models
