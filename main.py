import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataloader import Pointcloud, DynamicPointcloud, PointcloudNFrameSequenceDataset, PointcloudDataset
from model import PointMLPNFrames, PointTransformerNFrames
from trainer import Trainer
from evaluator import Evaluator

import wandb

# Stages to run:
# 0. WandB Setup
# 1. Data Setup
# 2. Model Setup
# 3. Load pretrained model
# 4. Training
# 5. Evaluation (Rollout)
# 6. Visualization

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
        "batch_size": 256,
        "hidden_dim": 128,
        "learning_rate": 1e-3,
        "scheduler_step_size": 50,
        "scheduler_gamma": 0.95,
        "save_model_checkpoint_iter": 500,
        "dataset_name": "orbiting_dataset_corrected"
    }
    
    print("Initializing wandb...")
    wandb.init(project='PointTransformerNFrames_tmp', config=config, dir='/work/williamb/gnode_wandb')
    print("wandb initialized")

    # 1. Data Setup
    # root_dir = f"data/{config['dataset_name']}"  # Contains 'train' and 'test' folders
    root_dir = f"/scratch/shared/beegfs/williamb/gnode/data/{config['dataset_name']}"  # Contains 'train' and 'test' folders

    input_sequence_length = config["input_sequence_length"]
    output_sequence_length = config["output_sequence_length"]
    epochs = config["epochs"]


    # Create train dataset and dataloader
    train_dataset = PointcloudDataset(
        root_dir,
        split='train',
        input_sequence_length=input_sequence_length,
        output_sequence_length=output_sequence_length,
        use_position=True,
        use_object_id=True,
        use_vertex_id=True
    )
    
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=config["batch_size"], 
        shuffle=True
    )

    # Create test dataset and dataloader
    test_dataset = PointcloudDataset(
        root_dir,
        split='test',
        input_sequence_length=input_sequence_length,
        output_sequence_length=output_sequence_length,
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

    # model = PointMLPNFrames(
    #     input_dim=input_dim, 
    #     output_dim=output_dim, 
    #     hidden_dim=config["hidden_dim"],
    #     input_sequence_length=input_sequence_length,
    #     output_sequence_length=output_sequence_length
    # )

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
    # Load the first test sequence
    test_sequence_path = "data/orbiting_dataset_corrected/test/obj_sequence_91"  # Adjust path as needed

    # Initialize and load the pointcloud
    gt_pc = DynamicPointcloud()
    gt_pc.load_obj_sequence(test_sequence_path)

    # Create and display the animation
    fig = gt_pc.to_plotly_figure()
    fig.show()

    html_str_gt = fig.to_html(full_html=False, include_plotlyjs='cdn')
    wandb.log({
        "ground_truth_visualization": wandb.Html(html_str_gt, inject=False),
    })


    # 4. Training
    model.train()
    if run_training:   
        trainer = Trainer(model, criterion, optimizer, scheduler, device, model_checkpoint_folder, model_checkpoint_name, config["save_model_checkpoint_iter"])
        trainer.train(train_dataloader, epochs=epochs)


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
    
    # 6. Visualization
    fig_pred = pred_dyn_pc.to_plotly_figure()
    fig_pred.show()

    # Create comparison visualization
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

if __name__ == "__main__":
    main()