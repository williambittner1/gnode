from dataloader import Pointcloud, DynamicPointcloud, PointcloudSequenceDataset
from model import PointMLP
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np


if __name__ == "__main__":
    input_folder = "data/obj_sequence1"

    gt_dyn_pc = DynamicPointcloud()
    gt_dyn_pc.load_obj_sequence(input_folder)

    # fig = gt_dyn_pc.to_plotly_figure()
    # fig.show()

    dataset = PointcloudSequenceDataset(gt_dyn_pc, use_position=True, use_object_id=True, use_vertex_id=True)
    dataloader = DataLoader(dataset, batch_size=512, shuffle=True)
    
    # get input dimension
    X_t, X_tp1 = dataset[0]
    input_dim = X_t.shape[1]
    output_dim = X_tp1.shape[1]

    print(f"Input dimension: {input_dim}")
    print(f"Output dimension: {output_dim}")

    model = PointMLP(input_dim, output_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.9)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(10000):
        for X_t, Y_t in dataloader:
            # X_t: (B, N, input_dim) where B=1 if no batching across samples.
            # Y_t: (B, N, 3)

            # If B=1, we can squeeze the batch dimension
            X_t = X_t.squeeze(0)  # (N, input_dim)
            Y_t = Y_t.squeeze(0)  # (N, 3)

            pred = model(X_t)  # (N,3)
            loss = criterion(pred, Y_t)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()
        print(f"Epoch {epoch}: Loss {loss.item():.4f}")

    # model rollout for 100 timesteps from the first frames

    # After training loop
    model.eval()

    # Get initial data from the first sample in the dataset
    # dataset[0] corresponds to frames (t=1 and t=2): X_t=frame 1, Y_t=frame 2
    X_current, Y_current = dataset[0]

    # X_current shape: (N, input_dim)
    # If position, object_id, vertex_id are all used:
    # input_dim = 5:
    #  - columns 0:3 -> x,y,z
    #  - column 3 -> object_id
    #  - column 4 -> vertex_id
    positions = X_current[:, :3].clone().detach()
    col = 3
    if dataset.use_object_id:
        obj_id = X_current[:, col:col+1].int().clone().detach()
        col += 1
    else:
        obj_id = None
    if dataset.use_vertex_id:
        vert_id = X_current[:, col:col+1].int().clone().detach()
    else:
        vert_id = None

    # We'll autoregress for 100 future frames
    num_steps = 100

    # predicted_sequence will store positions for frame 1 (known) and beyond
    predicted_sequence = [positions.numpy()]  # frame 1 positions

    # X_in is the current input to model
    X_in = X_current.clone().detach()

    for step in range(num_steps):
        with torch.no_grad():
            # Model predicts next positions (N,3)
            pred_positions = model(X_in)  
            
        # Store predicted positions
        predicted_sequence.append(pred_positions.numpy())

        # Update X_in for next step:
        # Construct new input with predicted positions, and original ids
        new_input_parts = [pred_positions]
        if obj_id is not None:
            new_input_parts.append(obj_id.float())
        if vert_id is not None:
            new_input_parts.append(vert_id.float())
        X_in = torch.cat(new_input_parts, dim=1)

    # Now we have predicted_sequence:
    # predicted_sequence[0] = frame 1 (original)
    # predicted_sequence[1] = frame 2 (predicted)
    # ...
    # predicted_sequence[num_steps] = frame (1+num_steps)

    # Construct a new DynamicPointcloud for visualization
    pred_dyn_pc = DynamicPointcloud()

    # Fill pred_dyn_pc.frames with Pointcloud instances
    # Use the same object_id and vertex_id for all frames
    for i, pos in enumerate(predicted_sequence):
        frame_number = i + 1
        pc = Pointcloud()
        pc.positions = pos
        if obj_id is not None:
            pc.object_id = obj_id.numpy().flatten()
        else:
            pc.object_id = np.zeros(pc.positions.shape[0], dtype=int)
        if vert_id is not None:
            pc.vertex_id = vert_id.numpy().flatten()
        else:
            pc.vertex_id = np.arange(pc.positions.shape[0], dtype=int)

        pred_dyn_pc.frames[frame_number] = pc

    # Visualize the predicted sequence
    fig = pred_dyn_pc.to_plotly_figure()
    fig.show()