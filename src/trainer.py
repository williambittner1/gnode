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

    def train_epoch(self, dataloader) -> float:
        """Train model for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        for X, y in dataloader:

            X = X.to(self.device)
            y = y.to(self.device)
            
            self.optimizer.zero_grad()
            pred = self.model(X)
            loss = self.criterion(pred, y)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(dataloader)

