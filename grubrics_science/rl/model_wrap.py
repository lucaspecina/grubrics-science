"""Wrapper for trainable GRubrics model (Qwen)."""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List
import numpy as np

from ..llm.client import QwenClient


class GRubricsModelWrapper:
    """Wrapper for Qwen model used as GRubrics."""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        device: str = "cuda",
        dtype: str = "bfloat16"
    ):
        """
        Initialize GRubrics model wrapper.
        
        Args:
            model_name: HuggingFace model name
            device: Device to use
            dtype: Data type
        """
        self.client = QwenClient(model_name=model_name, device=device, dtype=dtype)
        self.model = self.client.get_model()
        self.tokenizer = self.client.get_tokenizer()
        self.device = device
        self.dtype = getattr(torch, dtype) if hasattr(torch, dtype) else torch.bfloat16
    
    def generate_rubrics(
        self,
        prompt: str,
        num_samples: int = 4,
        max_new_tokens: int = 512,
        temperature: float = 1.0,
        top_k: int = 50
    ) -> List[str]:
        """
        Generate multiple rubric samples.
        
        Args:
            prompt: Input prompt
            num_samples: Number of samples to generate
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
        
        Returns:
            List of generated rubric texts
        """
        rubrics = []
        
        for _ in range(num_samples):
            rubric = self.client.generate_sync(
                prompt=prompt,
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k
            )
            rubrics.append(rubric.strip())
        
        return rubrics
    
    def compute_logprobs(
        self,
        prompt: str,
        completion: str
    ) -> Tuple[torch.Tensor, int]:
        """
        Compute log probabilities for a completion given a prompt.
        
        Args:
            prompt: Input prompt
            completion: Generated completion
        
        Returns:
            Tuple of (logprobs tensor, prompt_length)
        """
        # Tokenize prompt + completion
        full_text = prompt + completion
        tokens = self.tokenizer.encode(full_text, return_tensors="pt").to(self.device)
        prompt_tokens = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        prompt_length = prompt_tokens.shape[1]
        
        # Forward pass
        self.model.eval()
        with torch.no_grad():
            logits = self.model(tokens).logits
        
        # Compute log probs
        log_probs = torch.log_softmax(logits, dim=-1)
        
        # Extract log probs for completion tokens
        targets = tokens[:, 1:]  # Shift by 1 for next-token prediction
        selected_logprobs = log_probs[:, :-1].gather(2, targets.unsqueeze(-1)).squeeze(-1)
        
        # Return only completion part (after prompt)
        completion_logprobs = selected_logprobs[:, prompt_length - 1:]
        
        return completion_logprobs.squeeze(0), prompt_length
    
    def get_optimizer(self, learning_rate: float = 1e-5):
        """Get optimizer for the model."""
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
    
    def train_mode(self):
        """Set model to training mode."""
        self.model.train()
    
    def eval_mode(self):
        """Set model to evaluation mode."""
        self.model.eval()
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'tokenizer': self.tokenizer,
        }, path)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

