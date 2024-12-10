import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataloader import Pointcloud, DynamicPointcloud, PointcloudNFrameSequenceDataset
from model import PointMLPNFrames  
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
run_load_pretrained_model = True
# 4. Training
run_training = False
# 5. Evaluation (Rollout)
# run_evaluation = True
# 6. Visualization
# run_visualization = True


def main():
    
    # 0. WandB Setup
    config = {
        "input_sequence_length": 5,
        "output_sequence_length": 3,
        "epochs": 1_000,
        "batch_size": 512,
        "hidden_dim": 128,
        "learning_rate": 1e-3,
        "scheduler_step_size": 1000,
        "scheduler_gamma": 0.9,
        "save_model_checkpoint_iter": 100
    }
    
    wandb.init(project='gnode_trainer', config=config, dir='/work/williamb/gnode_wandb')


    # 1. Data Setup
    input_folder = "data/obj_sequence1"

    gt_dyn_pc = DynamicPointcloud()
    gt_dyn_pc.load_obj_sequence(input_folder)

    input_sequence_length = config["input_sequence_length"]
    output_sequence_length = config["output_sequence_length"]
    epochs = config["epochs"]

    dataset = PointcloudNFrameSequenceDataset(
        gt_dyn_pc, 
        input_sequence_length=input_sequence_length,
        output_sequence_length=output_sequence_length,
        use_position=True, 
        use_object_id=True, 
        use_vertex_id=True
    )

    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)


    # 2. Model Setup
    X_t, Y_t = dataset[0]
    input_dim = X_t.shape[-1]
    output_dim = Y_t.shape[-1]

    model = PointMLPNFrames(
        input_dim=input_dim, 
        output_dim=output_dim, 
        hidden_dim=config["hidden_dim"],
        input_sequence_length=input_sequence_length,
        output_sequence_length=output_sequence_length
    )

    model_checkpoint_name = f"{model.__class__.__name__}_input_seq{input_sequence_length}_output_seq{output_sequence_length}_{epochs}_epochs"
    model_checkpoint_folder = f"model_checkpoint/{model_checkpoint_name}"
    os.makedirs(model_checkpoint_folder, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config["scheduler_step_size"], gamma=config["scheduler_gamma"])
    criterion = nn.MSELoss()


    # 3. Load pretrained model
    if run_load_pretrained_model:
        model.load_state_dict(torch.load(f"{model_checkpoint_folder}/{model_checkpoint_name}.pth"))
    

    # 4. Training
    if run_training:   
        trainer = Trainer(model, criterion, optimizer, scheduler, device, model_checkpoint_folder, model_checkpoint_name, config["save_model_checkpoint_iter"])
        trainer.train(dataloader, epochs=epochs)


    # 5. Evaluation (Rollout)
    model.eval()

    evaluator = Evaluator(model, device)
    initial_X = dataset[0][0]  # dataset[0] gives (X_t, Y_t), [0] takes X_t
    
    pred_dyn_pc = evaluator.rollout(
        initial_X,
        use_position=True,
        use_object_id=dataset.use_object_id,
        use_vertex_id=dataset.use_vertex_id,
        rollout_length=100
    )
    
    # 6. Visualization
    fig = pred_dyn_pc.to_plotly_figure()
    fig.show()

    if run_training:   
        html_str = fig.to_html(full_html=False, include_plotlyjs='cdn')
        wandb.log({
            "prediction_visualization": wandb.Html(html_str, inject=False)
        })

    wandb.finish()

if __name__ == "__main__":
    main()