import torch
import wandb
from tqdm import tqdm
from typing import Dict, Any

from dataloader import get_positional_encoding

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


    def add_noise_to_input(self, X):
        """Add Gaussian noise to input features."""
        # Get data statistics
        data_std = torch.std(X, dim=0)
        
        # Create noise with same shape as input
        noise = torch.randn_like(X)
        
        # Scale noise by data std and noise factor
        if self.config.num_freq_bands > 0:
            # If using positional encoding, only add noise to the first 3 dimensions (xyz positions)
            noise_scale = data_std[:3] * self.config.noise_scale_factor
            X_noisy = X.clone()
            X_noisy[..., :3] = X[..., :3] + noise[..., :3] * noise_scale
        else:
            noise_scale = data_std * self.config.noise_scale_factor
            X_noisy = X + noise * noise_scale
            
        return X_noisy


    def train_epoch(self, dataloader) -> float:
        """Train model for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        for X, y in dataloader:

            X = X.to(self.device)
            y = y.to(self.device)
            
            if self.config.noise_enabled:
                X_noisy = self.add_noise_to_input(X)
                self.optimizer.zero_grad()
                pred = self.model(X_noisy)
            else:
                self.optimizer.zero_grad()
                pred = self.model(X)
                
            loss = self.criterion(pred, y)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(dataloader)

