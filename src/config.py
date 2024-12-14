from dataclasses import dataclass
import os

@dataclass
class Config:
    # Training parameters
    input_sequence_length: int = 10
    output_sequence_length: int = 20
    epochs: int = 1000
    batch_size: int = 256
    hidden_dim: int = 128
    learning_rate: float = 1e-3
    scheduler_step_size: int = 50
    scheduler_gamma: float = 0.95
    
    # Model parameters
    num_heads: int = 4
    num_transformer_layers: int = 2
    dropout: float = 0.1
    
    # Checkpointing
    save_model_checkpoint_iter: int = 2
    model_checkpoint_dir: str = "model_checkpoint"
    
    # Dataset
    dataset_name: str = "mini_damped_orbit_h5"
    dataset_root: str = "data/mini_damped_orbit_h5"
    # dataset_root: str = "/scratch/shared/beegfs/williamb/gnode/data/medium_damped_orbit_h5"

    # Pretrained model
    run_load_pretrained_model: bool = False
    pretrained_model_path: str = "model_checkpoint/medium_damped_orbit_h5/test_model.pth"

    # Visualization
    viz_iter: int = 2
    rollout_length: int = 500
    viewport_size: int = 12
    
    # Wandb
    wandb_project: str = "debug"
    wandb_dir: str = "/work/williamb/gnode_wandb"

    @property
    def model_checkpoint_path(self):
        return os.path.join(self.model_checkpoint_dir, self.dataset_name)