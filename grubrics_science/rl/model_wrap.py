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
        Generate multiple rubric samples (for inference/eval).
        
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
    
    @torch.no_grad()
    def sample_rubric_tokens(
        self,
        prompt: str,
        max_new_tokens: int = 2048,
        temperature: float = 1.0,
        top_k: int = 50
    ) -> Tuple[torch.Tensor, int, str]:
        """
        Sample rubric tokens (for training).
        
        Returns:
            Tuple of (full_token_ids, prompt_length, generated_text)
        """
        # Tokenize prompt
        prompt_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        prompt_length = prompt_ids.shape[1]
        
        # Generate tokens
        self.model.eval()
        with torch.no_grad():
            generated_ids = self.model.generate(
                prompt_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Extract generated part
        generated_token_ids = generated_ids[0, prompt_length:]
        generated_text = self.tokenizer.decode(generated_token_ids, skip_special_tokens=True)
        
        # Return full sequence (prompt + generated)
        return generated_ids[0], prompt_length, generated_text
    
    def compute_logprobs_per_token(
        self,
        token_ids: torch.Tensor,
        prompt_length: int,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute log probabilities per token (following nanochat pattern).
        
        In nanochat: logp = -model(inputs, targets, loss_reduction='none')
        Since we use HuggingFace Qwen, we compute it manually.
        
        Args:
            token_ids: Full token sequence (prompt + generated), shape [seq_len]
            prompt_length: Length of prompt (to exclude from logprob)
            mask: Optional mask (1 for valid tokens, 0 to ignore), shape [seq_len]
        
        Returns:
            Log probabilities per token for generated part, shape [gen_len]
            (following nanochat: returns per-token logprobs, not sum)
        """
        # Ensure model is in train mode for gradients
        self.model.train()
        
        # Prepare inputs and targets (following nanochat pattern)
        # inputs = all tokens except last, targets = all tokens shifted by 1
        inputs = token_ids[:-1].unsqueeze(0)  # (1, T-1)
        targets = token_ids[1:]  # (T-1,)
        
        # Forward pass to get logits (WITH gradients)
        outputs = self.model(inputs)
        logits = outputs.logits.squeeze(0).float()  # (T-1, vocab_size)
        
        # Compute log probs (following nanochat: logp = -cross_entropy_loss)
        # But we'll compute it manually: log_softmax then gather
        log_probs = torch.log_softmax(logits, dim=-1)  # (T-1, vocab_size)
        
        # Select logprobs for actual target tokens
        selected_logprobs = log_probs.gather(1, targets.unsqueeze(-1)).squeeze(-1)  # (T-1,)
        
        # Extract only generated part (after prompt)
        # Note: prompt_length-1 because we shifted by 1 for targets
        generated_logprobs = selected_logprobs[prompt_length - 1:]
        
        # Apply mask if provided (for ignoring certain tokens)
        if mask is not None:
            gen_mask = mask[prompt_length:]  # mask for generated tokens
            generated_logprobs = generated_logprobs * gen_mask
        
        return generated_logprobs  # Return per-token, not sum (following nanochat)
    
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

