import torch
import wandb
from tqdm import tqdm
from typing import Dict, Any

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        device: torch.device,
        config: Any
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config


    def add_noise_to_input(self, X: torch.Tensor) -> torch.Tensor:
        """
        Add Gaussian noise to position coordinates while preserving other features.
        Args:       X: Input tensor of shape (..., N) where first 3 dimensions are positions
        Returns:    X_noisy: Input tensor with noise added to position coordinates
        """
        # Scale noise based on data magnitude
        noise_scale = 0.01 * torch.std(X[..., :3])  # Only add noise to position coordinates
        noise = torch.randn_like(X[..., :3]) * noise_scale
        
        # Create noisy input while preserving non-position features
        X_noisy = X.clone()
        X_noisy[..., :3] = X[..., :3] + noise
        
        return X_noisy
    

    def train_epoch(self, dataloader) -> float:
        """Train model for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        for X, y in dataloader:

            X = X.to(self.device)
            y = y.to(self.device)
            
            X_noisy = self.add_noise_to_input(X)

            self.optimizer.zero_grad()
            pred = self.model(X)
            loss = self.criterion(pred, y)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(dataloader)

