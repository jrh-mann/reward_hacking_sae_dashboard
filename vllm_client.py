"""
vLLM Client for querying the gpt-oss-20b model server.

This module provides a simple interface to interact with the vLLM server
running the gpt-oss-20b model via OpenAI-compatible API.
"""

import requests
from typing import List, Dict, Any, Optional, Union


class VLLMClient:
    """Client for interacting with vLLM server."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize the vLLM client.
        
        Args:
            base_url: Base URL of the vLLM server (default: http://localhost:8000)
        """
        self.base_url = base_url.rstrip('/')
        self.completions_url = f"{self.base_url}/v1/completions"
        self.chat_url = f"{self.base_url}/v1/chat/completions"
        self.models_url = f"{self.base_url}/v1/models"
    
    def generate(
        self,
        prompt: Union[str, List[str]],
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        n: int = 1,
        stop: Optional[Union[str, List[str]]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate completions for the given prompt(s).
        
        Args:
            prompt: Input prompt(s) to generate completions for
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 = deterministic)
            top_p: Nucleus sampling parameter
            n: Number of completions to generate
            stop: Stop sequences
            **kwargs: Additional parameters to pass to the API
        
        Returns:
            Dict containing the API response with generated text
        
        Example:
            >>> client = VLLMClient()
            >>> result = client.generate("Once upon a time", max_tokens=50)
            >>> print(result['choices'][0]['text'])
        """
        data = {
            "model": "openai/gpt-oss-20b",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "n": n,
            **kwargs
        }
        
        if stop is not None:
            data["stop"] = stop
        
        response = requests.post(
            self.completions_url,
            json=data,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        return response.json()
    
    def get_text(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """
        Convenience method to get just the generated text.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional parameters
        
        Returns:
            Generated text string
        
        Example:
            >>> client = VLLMClient()
            >>> text = client.get_text("The meaning of life is")
            >>> print(text)
        """
        result = self.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            n=1,
            **kwargs
        )
        return result['choices'][0]['text']
    
    def batch_generate(
        self,
        prompts: List[str],
        max_tokens: int = 100,
        temperature: float = 0.7,
        **kwargs
    ) -> List[str]:
        """
        Generate completions for multiple prompts.
        
        Args:
            prompts: List of input prompts
            max_tokens: Maximum number of tokens to generate per prompt
            temperature: Sampling temperature
            **kwargs: Additional parameters
        
        Returns:
            List of generated text strings
        
        Example:
            >>> client = VLLMClient()
            >>> prompts = ["Hello", "Goodbye", "How are you"]
            >>> results = client.batch_generate(prompts)
            >>> for prompt, result in zip(prompts, results):
            ...     print(f"{prompt} -> {result}")
        """
        result = self.generate(
            prompt=prompts,
            max_tokens=max_tokens,
            temperature=temperature,
            n=1,
            **kwargs
        )
        return [choice['text'] for choice in result['choices']]
    
    def list_models(self) -> Dict[str, Any]:
        """
        List available models on the server.
        
        Returns:
            Dict containing model information
        """
        response = requests.get(self.models_url)
        response.raise_for_status()
        return response.json()
    
    def health_check(self) -> bool:
        """
        Check if the server is running and healthy.
        
        Returns:
            True if server is healthy, False otherwise
        """
        try:
            response = requests.get(f"{self.base_url}/health")
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False


# Standalone functions for simple usage
def generate_text(
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 0.7,
    base_url: str = "http://localhost:8000"
) -> str:
    """
    Simple function to generate text from a prompt.
    
    Args:
        prompt: Input prompt
        max_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        base_url: URL of the vLLM server
    
    Returns:
        Generated text
    
    Example:
        >>> text = generate_text("The quick brown fox")
        >>> print(text)
    """
    client = VLLMClient(base_url=base_url)
    return client.get_text(prompt, max_tokens=max_tokens, temperature=temperature)


if __name__ == "__main__":
    # Example usage
    print("vLLM Client Example Usage")
    print("=" * 50)
    
    # Initialize client
    client = VLLMClient(base_url="http://localhost:8000")
    
    # Check if server is running
    print("\n1. Health Check:")
    if client.health_check():
        print("   ✓ Server is running")
    else:
        print("   ✗ Server is not running. Start it with: bash start_vllm_server.sh")
        exit(1)
    
    # List models
    print("\n2. Available Models:")
    try:
        models = client.list_models()
        print(f"   {models}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Generate single completion
    print("\n3. Single Completion:")
    prompt = "The quick brown fox"
    print(f"   Prompt: '{prompt}'")
    try:
        text = client.get_text(prompt, max_tokens=30, temperature=0.7)
        print(f"   Generated: '{text}'")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Batch generation
    print("\n4. Batch Generation:")
    prompts = [
        "Once upon a time",
        "In a galaxy far away",
        "The meaning of life is"
    ]
    try:
        results = client.batch_generate(prompts, max_tokens=20)
        for p, r in zip(prompts, results):
            print(f"   '{p}' -> '{r}'")
    except Exception as e:
        print(f"   Error: {e}")
    
    print("\n" + "=" * 50)
    print("Done!")

