"""LLM client abstractions.

Supports:
- Qwen (for trainable GRubrics model)
- Azure OpenAI (for fixed Judge and Answer Policy)
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
import os

# Load environment variables from .env if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from openai import AsyncOpenAI, AsyncAzureOpenAI
except ImportError:
    AsyncOpenAI = None
    AsyncAzureOpenAI = None

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
except ImportError:
    AutoModelForCausalLM = None
    AutoTokenizer = None
    torch = None


class LLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 1.0,
        top_k: int = 50,
        **kwargs
    ) -> str:
        """Generate text from prompt."""
        pass
    
    @abstractmethod
    def generate_sync(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 1.0,
        top_k: int = 50,
        **kwargs
    ) -> str:
        """Synchronous version of generate (for training loop)."""
        pass


class AzureOpenAIClient(LLMClient):
    """Azure OpenAI client for Judge and Answer Policy."""
    
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        use_azure: bool = True,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        api_version: Optional[str] = None
    ):
        if AsyncOpenAI is None or AsyncAzureOpenAI is None:
            raise ImportError("openai package required. Install with: pip install openai")
        
        self.model = model
        self.use_azure = use_azure
        
        if use_azure:
            api_key = api_key or os.environ.get("AZURE_API_KEY", "")
            api_base = api_base or os.environ.get("AZURE_API_BASE", "")
            api_version = api_version or os.environ.get("AZURE_API_VERSION", "2024-12-01-preview")
            
            if not api_key:
                raise ValueError("AZURE_API_KEY not set")
            
            self.client = AsyncAzureOpenAI(
                api_key=api_key,
                azure_endpoint=api_base.rstrip('/'),
                api_version=api_version
            )
        else:
            api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not set")
            self.client = AsyncOpenAI(api_key=api_key)
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 1.0,
        top_k: int = 50,
        **kwargs
    ) -> str:
        """Generate text using Azure OpenAI."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # GPT-5+ only supports temperature=1; omit it to use the default.
        params = dict(
            model=self.model,
            messages=messages,
            max_completion_tokens=max_tokens,
            **kwargs,
        )
        if temperature != 1.0:
            params["temperature"] = temperature
        response = await self.client.chat.completions.create(**params)
        return response.choices[0].message.content
    
    def generate_sync(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 1.0,
        top_k: int = 50,
        **kwargs
    ) -> str:
        """Synchronous version (uses async under the hood)."""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(
            self.generate(prompt, system_prompt, max_tokens, temperature, top_k, **kwargs)
        )


class QwenClient(LLMClient):
    """Qwen client for trainable GRubrics model."""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        device: str = "cuda",
        dtype: str = "bfloat16"
    ):
        if AutoModelForCausalLM is None:
            raise ImportError("transformers package required. Install with: pip install transformers")
        if torch is None:
            raise ImportError("torch package required. Install with: pip install torch")
        
        self.model_name = model_name
        self.device = device
        self.dtype = getattr(torch, dtype) if hasattr(torch, dtype) else torch.bfloat16
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=self.dtype,
            device_map=device
        )
        self.model.eval()  # Start in eval mode
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 1.0,
        top_k: int = 50,
        **kwargs
    ) -> str:
        """Generate text using Qwen (async wrapper)."""
        return self.generate_sync(prompt, system_prompt, max_tokens, temperature, top_k, **kwargs)
    
    def generate_sync(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 1.0,
        top_k: int = 50,
        **kwargs
    ) -> str:
        """Generate text using Qwen."""
        # Combine system prompt and user prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        else:
            full_prompt = prompt
        
        # Tokenize
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                do_sample=temperature > 0,
                **kwargs
            )
        
        # Decode only the generated part
        generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    def get_model(self):
        """Get the underlying model for training."""
        return self.model
    
    def get_tokenizer(self):
        """Get the tokenizer."""
        return self.tokenizer

