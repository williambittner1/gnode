from dataclasses import dataclass
import os

@dataclass
class Config:
    # Training parameters
    input_sequence_length: int = 5
    output_sequence_length: int = 5
    epochs: int = 5000
    batch_size: int = 2048
    hidden_dim: int = 128
    learning_rate: float = 1e-4
    scheduler_step_size: int = 50
    scheduler_gamma: float = 0.95
    
    # Model parameters
    num_heads: int = 4
    num_transformer_layers: int = 2
    dropout: float = 0.1
    
    # Feature parameters
    use_position: bool = True
    use_object_id: bool = False
    use_vertex_id: bool = False
    use_initial_position: bool = True
    num_freq_bands: int = 0  # 0 means no frequency encoding, >0 specifies number of bands
    
    # Checkpointing
    save_model_checkpoint_iter: int = 500
    model_checkpoint_dir: str = "../model_checkpoint"
    
    # Dataset
    dataset_name: str = "medium"
    slurm: bool = True
    
    @property
    def dataset_root(self): 
        if self.slurm == True:
            return f"/scratch/shared/beegfs/williamb/gnode/data/{self.dataset_name}"
        else:
            return f"/users/williamb/dev/gnode/data/{self.dataset_name}"

    # Pretrained model
    run_load_pretrained_model: bool = False
    pretrained_model_path: str = "model_checkpoint/medium_damped_orbit_h5/test_model.pth"

    # Visualization and Metrics
    viz_iter: int = 50  # How often to log visualizations
    extrapolation_metrics_iter: int = 1  # How often to log metrics
    rollout_length: int = 200
    viewport_size: int = 12
    
    # Wandb
    wandb_project: str = "PointcloudTransformer"
    wandb_dir: str = "/work/williamb/gnode_wandb"

    @property
    def model_checkpoint_path(self):
        return os.path.join(self.model_checkpoint_dir, self.dataset_name)

    # Add these to your existing Config class
    noise_enabled: bool = True
    noise_scale_factor: float = 0.01  # 1% of data std