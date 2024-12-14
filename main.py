import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# from dataloader import Pointcloud, DynamicPointcloud, PointcloudNFrameSequenceDataset, PointcloudDataset
from dataloader import PointcloudH5Dataset, DynamicPointcloud
from model import PointMLPNFrames, PointTransformerNFrames, PointTransformerGNN
from trainer import Trainer
from evaluator import Evaluator

import wandb

# Stages to run:

# 0. WandB Setup
# run_wandb_setup = True
# 1. Data Setup
# run_data_setup = True
# 2. Model Setup
# run_model_setup = True
# 3. Load pretrained model
run_load_pretrained_model = False
# 4. Training
run_training = True
# 5. Evaluation (Rollout)
# run_evaluation = True
# 6. Visualization
# run_visualization = True


def main():
    
    # 0. WandB Setup
    config = {
        "input_sequence_length": 10,
        "output_sequence_length": 1,
        "epochs": 1000,
        "batch_size": 128,
        "hidden_dim": 128,
        "learning_rate": 1e-3,
        "scheduler_step_size": 50,
        "scheduler_gamma": 0.95,
        "save_model_checkpoint_iter": 500,
        "dataset_name": "medium_damped_orbit_h5",
        "rollout_pred_viz_iter": 100,  # Log visualization every N epochs
        "rollout_length": 300  # Number of steps to predict in visualization
    }
    
    print("Initializing wandb...")
    wandb.init(project='PointTransformer', config=config, dir='/work/williamb/gnode_wandb')
    print("wandb initialized")

    # 1. Data Setup
    # root_dir = f"data/{config['dataset_name']}"  # debug machine training
    root_dir = f"/scratch/shared/beegfs/williamb/gnode/data/{config['dataset_name']}"  # athena-cluster training

    input_sequence_length = config["input_sequence_length"]
    output_sequence_length = config["output_sequence_length"]
    epochs = config["epochs"]


    train_dataset = PointcloudH5Dataset(
        root_dir=root_dir,
        split='train',
        input_sequence_length=10,
        output_sequence_length=1,
        use_position=True,
        use_object_id=True,
        use_vertex_id=True
    )

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=config["batch_size"], 
        shuffle=True
    )

    test_dataset = PointcloudH5Dataset(
        root_dir=root_dir,
        split='test',
        input_sequence_length=10,
        output_sequence_length=1,
        use_position=True,
        use_object_id=True,
        use_vertex_id=True
    )
    
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=config["batch_size"], 
        shuffle=False
    )


    # 2. Model Setup
    X_t, Y_t = train_dataset[0]
    input_dim = X_t.shape[-1]
    output_dim = Y_t.shape[-1]


    model = PointTransformerNFrames(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=config["hidden_dim"],
        input_sequence_length=input_sequence_length,
        output_sequence_length=output_sequence_length
    )

    model_checkpoint_name = f"{model.__class__.__name__}_input_seq{input_sequence_length}_output_seq{output_sequence_length}_{epochs}_epochs"
    model_checkpoint_folder = f"model_checkpoint/{config['dataset_name']}/{model_checkpoint_name}"
    os.makedirs(model_checkpoint_folder, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config["scheduler_step_size"], gamma=config["scheduler_gamma"])
    criterion = nn.MSELoss()


    # 3. Load pretrained model
    if run_load_pretrained_model:
        model.load_state_dict(torch.load(f"{model_checkpoint_folder}/{model_checkpoint_name}.pth"))
    

    # 3.5 GT Visualization
    test_sequence_path = "data/mini_damped_orbit_h5/test/sequence_9.h5"
    dyn_pc = DynamicPointcloud()
    dyn_pc.load_h5_sequence(test_sequence_path)
    dyn_pc.log_visualization_to_wandb("ground_truth_visualization")


    # 4. Training
    model.train()
    if run_training:   

        trainer = Trainer(model, criterion, optimizer, scheduler, device, model_checkpoint_folder, model_checkpoint_name, config["save_model_checkpoint_iter"])

        progress_bar = tqdm(range(epochs), desc="Training", unit="epoch")
        for epoch in progress_bar:
            avg_loss = trainer.train_one_epoch(train_dataloader)
            current_lr = trainer.optimizer.param_groups[0]['lr']
            # Log metrics to wandb
            wandb.log({
                "epoch": epoch,
                "train_loss": avg_loss,
                "learning_rate": current_lr
            })
            
            if trainer.scheduler:
                trainer.scheduler.step()

            # Intermediate checkpoints
            if (epoch + 1) % trainer.save_model_iter == 0:
                torch.save(trainer.model.state_dict(), f"{trainer.model_checkpoint_folder}/{trainer.model_checkpoint_name}_{epoch + 1}.pth")
                print(f"Intermediate checkpoint saved at epoch {epoch + 1}") 

            # Final checkpoint
            if epoch == epochs - 1:
                torch.save(trainer.model.state_dict(), f"{trainer.model_checkpoint_folder}/{trainer.model_checkpoint_name}.pth")
                print(f"Final checkpoint saved at epoch {epoch + 1}")

            # Log visualization at specified intervals
            if (epoch + 1) % config["rollout_pred_viz_iter"] == 0:
                log_rollout_visualization(
                    model, 
                    test_dataset, 
                    device, 
                    config["rollout_length"]
                )

            # Update progress bar description with current metrics
            progress_bar.set_description(
                f"Training [Loss: {avg_loss:.7f} | LR: {current_lr:.3e}]"
            )
        



    # 5. Evaluation (Rollout)
    model.eval()

    evaluator = Evaluator(model, device)
    initial_X = test_dataset[0][0]  # dataset[0] gives (X_t, Y_t), [0] takes X_t
    
    # Get ground truth sequence
    gt_dyn_pc = DynamicPointcloud()
    gt_dyn_pc.load_obj_sequence(test_sequence_path)
    

    pred_dyn_pc = evaluator.rollout(
        initial_X,
        use_position=True,
        use_object_id=test_dataset.use_object_id,
        use_vertex_id=test_dataset.use_vertex_id,
        rollout_length=200
    )
    
    fig_pred = pred_dyn_pc.to_plotly_figure()
    fig_pred.show()

    
    fig_comparison = gt_dyn_pc.create_comparison_figure(pred_dyn_pc)
    fig_comparison.show()


    if run_training:   
        html_str_pred = fig_pred.to_html(full_html=False, include_plotlyjs='cdn')
        html_str_gt_vs_pred = fig_comparison.to_html(full_html=False, include_plotlyjs='cdn')  
        wandb.log({
            "prediction_visualization": wandb.Html(html_str_pred, inject=False),
            "gt_vs_pred_visualization": wandb.Html(html_str_gt_vs_pred, inject=False)
        })


    wandb.finish()




def log_rollout_visualization(model, test_dataset, device, rollout_length):
    """
    Creates and logs visualization of model predictions on the first test sequence.
    
    Args:
        model: The trained model
        test_dataset: The test dataset
        device: The device to run predictions on
        rollout_length: Number of steps to predict into the future
    """
    model.eval()
    evaluator = Evaluator(model, device)
    
    # Get first test sequence
    initial_X = test_dataset[0][0]  # dataset[0] gives (X_t, Y_t), [0] takes X_t
    
    # Get ground truth sequence
    test_sequence_path = os.path.join(test_dataset.split_dir, test_dataset.sequence_files[0])
    gt_dyn_pc = DynamicPointcloud()
    gt_dyn_pc.load_h5_sequence(test_sequence_path)
    
    # Generate predictions
    pred_dyn_pc = evaluator.rollout(
        initial_X,
        use_position=True,
        use_object_id=test_dataset.use_object_id,
        use_vertex_id=test_dataset.use_vertex_id,
        rollout_length=rollout_length
    )
    
    # Create comparison visualization
    fig_comparison = gt_dyn_pc.create_comparison_figure(pred_dyn_pc)
    html_str = fig_comparison.to_html(full_html=False, include_plotlyjs='cdn')
    
    # Log to wandb
    wandb.log({
        "rollout_visualization": wandb.Html(html_str, inject=False), 
        "key": "rollout_visualization"
    })
    
    model.train()


if __name__ == "__main__":
    main()