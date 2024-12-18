# Standard Imports
import torch
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import numpy as np
from torch.cuda import memory_summary
import torch.cuda as cuda
import threading
import time
import open3d as o3d

# Local Imports
from config import Config
from trainer import Trainer
from visualization import log_dataset_visualizations,log_extrapolation_metrics, log_prediction_visualization
from dataloader import PointcloudH5Dataset
from model import PointTransformer
from pointcloud import DynamicPointcloud



def train(config: Config):
    
    # 0. Wandb Setup
    wandb.init(project=config.wandb_project, config=config.__dict__, dir=config.wandb_dir)
    

    # 1. Data Setup
    train_dataset = PointcloudH5Dataset(
        root_dir=config.dataset_root,
        split='train',
        input_sequence_length=config.input_sequence_length,
        output_sequence_length=config.output_sequence_length,
        use_position=config.use_position,
        use_object_id=config.use_object_id,
        use_vertex_id=config.use_vertex_id,
        use_initial_position=config.use_initial_position,
        num_freq_bands=config.num_freq_bands
    )
    
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True
    )
    
    test_dataset = PointcloudH5Dataset(
        root_dir=config.dataset_root,
        split='test',
        input_sequence_length=config.input_sequence_length,
        output_sequence_length=config.output_sequence_length,
        use_position=config.use_position,
        use_object_id=config.use_object_id,
        use_vertex_id=config.use_vertex_id,
        use_initial_position=config.use_initial_position,
        num_freq_bands=config.num_freq_bands
    )

    torch.cuda.empty_cache()
    

    # 2. Model Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PointTransformer(
        input_dim=train_dataset.feature_dim,
        output_dim=3,
        hidden_dim=config.hidden_dim,
        input_sequence_length=config.input_sequence_length,
        output_sequence_length=config.output_sequence_length,
        device=device
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=config.scheduler_step_size, 
        gamma=config.scheduler_gamma
    )
    criterion = torch.nn.MSELoss()
    

    # 3. Load pretrained model
    if config.run_load_pretrained_model:
        model.load_state_dict(torch.load(config.pretrained_model_path))


    # 4. Visualize Dataset-Samples
    visualize_train_dataset = False
    if visualize_train_dataset:
        log_dataset_visualizations(
            train_dataset, 
            num_sequences=2, 
            prefix="train"
        )
    visualize_test_dataset = False
    if visualize_test_dataset:
        log_dataset_visualizations(
            test_dataset, 
            num_sequences=1, 
            prefix="test"
        )


    # 5. Training
    torch.cuda.empty_cache()
    trainer = Trainer(model, criterion, optimizer, scheduler, device, config)
    
    progress_bar = tqdm(range(config.epochs), desc="Training")
    for epoch in progress_bar:

        avg_loss = trainer.train_epoch(train_dataloader)
        
        current_lr = scheduler.get_last_lr()[0]
        
        metrics = {
            "train_loss": avg_loss,
            "learning_rate": current_lr
        }

        # Get predictions if needed for either visualization or metrics
        if ((epoch + 1) % config.viz_iter == 0 or 
            (epoch + 1) % config.extrapolation_metrics_iter == 0 or 
            epoch == 50 or epoch == config.epochs - 1 or epoch == 0):
            
            initial_X = test_dataset[0][0]
            test_sequence_path = os.path.join(test_dataset.split_dir, test_dataset.sequence_files[0])
            gt_dyn_pc = DynamicPointcloud()
            gt_dyn_pc.load_h5_sequence(test_sequence_path)
            
            with torch.no_grad():
                pred_dyn_pc = model.rollout(
                    initial_X.to(device),
                    rollout_length=config.rollout_length,
                    config=config
                )
            
            # Get metrics if it's time
            if ((epoch + 1) % config.extrapolation_metrics_iter == 0 or 
                epoch == 50 or epoch == config.epochs - 1 or epoch == 0):
                extrapolation_metrics = log_extrapolation_metrics(gt_dyn_pc, pred_dyn_pc, epoch)
                metrics.update(extrapolation_metrics)

            # Log visualization if it's time
            # if ((epoch + 1) % config.viz_iter == 0 or 
            #     epoch == 50 or epoch == config.epochs - 1 or epoch == 0):
            #     log_prediction_visualization(gt_dyn_pc, pred_dyn_pc, epoch)

        # Log all metrics at once
        wandb.log({**metrics, "epoch": epoch}, step=epoch)

        # Save model checkpoint
        if (epoch + 1) % config.save_model_checkpoint_iter == 0 or epoch == config.epochs - 1:
            checkpoint_path = os.path.join(
                config.model_checkpoint_dir, 
                config.dataset_name,
                f"{model.__class__.__name__}_epoch{epoch}.pth"
            )
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved model checkpoint to {checkpoint_path}")


        progress_bar.set_description(
            f"Training [Loss: {avg_loss:.7f} | LR: {current_lr:.3e}]"
        )
        

if __name__ == "__main__":
    config = Config()
    train(config)