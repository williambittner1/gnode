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

# # Stages to run:
# # 1. Data Setup
# run_data_setup = True
# # 2. Model Setup
# run_model_setup = True
# # 3. Training
# run_training = True
# # 4. Save model
# run_save_model = True
# # 5. Load pretrained model
# run_load_pretrained_model = True
# # 6. Evaluation (Rollout)
# run_evaluation = True
# # 7. Visualization
# run_visualization = True


def main():

    # 0. Initialize wandb
    
    config = {
        "input_sequence_length": 5,
        "output_sequence_length": 3,
        "epochs": 100_000,
        "batch_size": 512,
        "hidden_dim": 128,
        "learning_rate": 1e-3,
        "scheduler_step_size": 1000,
        "scheduler_gamma": 0.9
    }
    
    wandb.init(project='gnode_trainer', config=config, dir='/work/williamb/gnode_wandb')




    # 1. Data Setup
    input_folder = "data/obj_sequence1"

    gt_dyn_pc = DynamicPointcloud()
    gt_dyn_pc.load_obj_sequence(input_folder)

    input_sequence_length = 5
    output_sequence_length = 3
    epochs = 100_000

    dataset = PointcloudNFrameSequenceDataset(
        gt_dyn_pc, 
        input_sequence_length=input_sequence_length,
        output_sequence_length=output_sequence_length,
        use_position=True, 
        use_object_id=True, 
        use_vertex_id=True
    )

    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)


    # 2. Model Setup
    X_t, Y_t = dataset[0]
    input_dim = X_t.shape[-1]
    output_dim = Y_t.shape[-1]

    model = PointMLPNFrames(
        input_dim=input_dim, 
        output_dim=output_dim, 
        hidden_dim=128,
        input_sequence_length=input_sequence_length,
        output_sequence_length=output_sequence_length
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)
    criterion = nn.MSELoss()


    # 3. Training
    trainer = Trainer(model, criterion, optimizer, scheduler, device)
    trainer.train(dataloader, epochs=epochs)


    # 4. Save model
    model_name = model.__class__.__name__
    torch.save(model.state_dict(), f"model_checkpoint/{model_name}_{epochs}_epochs_input_seq_{input_sequence_length}_output_seq_{output_sequence_length}.pth")

    # 5. Load pretrained model
    model.load_state_dict(torch.load(f"model_checkpoint/{model_name}_{epochs}_epochs_input_seq_{input_sequence_length}_output_seq_{output_sequence_length}.pth"))

    # 6. Evaluation (Rollout)
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
    
    # 7. Visualization
    fig = pred_dyn_pc.to_plotly_figure()
    fig.show()


if __name__ == "__main__":
    main()
