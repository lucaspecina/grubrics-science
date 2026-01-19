"""Logging utilities."""

import logging
from typing import Dict, Any, Optional


def setup_logging(level: int = logging.INFO):
    """Setup basic logging configuration."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance."""
    return logging.getLogger(name)


class DummyWandb:
    """Dummy wandb logger for when wandb is not used."""
    def log(self, *args, **kwargs):
        pass
    
    def finish(self):
        pass


def log_metrics(
    logger: logging.Logger,
    step: int,
    metrics: Dict[str, Any],
    wandb_logger: Optional[Any] = None
):
    """Log metrics to both logger and optionally wandb."""
    log_str = f"Step {step}"
    for key, value in metrics.items():
        log_str += f" | {key}: {value:.4f}" if isinstance(value, (int, float)) else f" | {key}: {value}"
    logger.info(log_str)
    
    if wandb_logger is not None:
        wandb_logger.log({"step": step, **metrics})

