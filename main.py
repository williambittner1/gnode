import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataloader import Pointcloud, DynamicPointcloud, PointcloudNFrameSequenceDataset
from model import PointMLPNFrames  
from trainer import Trainer
from evaluator import Evaluator


# Stages to run:
# 1. Data Setup
# 2. Model Setup
# 3. Training
# 4. Evaluation (Rollout)
# 5. Visualization

def main():

    # Data Setup
    input_folder = "data/obj_sequence1"

    gt_dyn_pc = DynamicPointcloud()
    gt_dyn_pc.load_obj_sequence(input_folder)

    sequence_length = 5
    epochs = 100

    dataset = PointcloudNFrameSequenceDataset(
        gt_dyn_pc, 
        sequence_length=sequence_length,
        use_position=True, 
        use_object_id=True, 
        use_vertex_id=True
    )

    dataloader = DataLoader(dataset, batch_size=512, shuffle=True)


    # Model Setup
    X_t, Y_t = dataset[0]
    input_dim = X_t.shape[-1]
    output_dim = Y_t.shape[-1]

    model = PointMLPNFrames(
        input_dim=input_dim, 
        output_dim=output_dim, 
        hidden_dim=128,
        sequence_length=sequence_length
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)
    criterion = nn.MSELoss()


    # Training
    trainer = Trainer(model, criterion, optimizer, scheduler, device)
    trainer.train(dataloader, epochs=epochs)


    # Save model
    model_name = model.__class__.__name__
    torch.save(model.state_dict(), f"model_checkpoint/{model_name}_{epochs}_epochs.pth")


    # Evaluation (Rollout)
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
    
    # Visualization
    fig = pred_dyn_pc.to_plotly_figure()
    fig.show()


if __name__ == "__main__":
    main()
