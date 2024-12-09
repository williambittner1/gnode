from dataloader import DynamicPointcloud, PointcloudSequenceDataset
from model import PointMLP
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


if __name__ == "__main__":
    input_folder = "data/obj_sequence1"

    gt_dyn_pc = DynamicPointcloud()
    gt_dyn_pc.load_obj_sequence(input_folder)

    fig = gt_dyn_pc.to_plotly_figure()
    fig.show()

    dataset = PointcloudSequenceDataset(gt_dyn_pc, use_position=True, use_object_id=True, use_vertex_id=True)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # get input dimension
    X_t, X_tp1 = dataset[0]
    input_dim = X_t.shape[1]
    output_dim = X_tp1.shape[1]

    print(f"Input dimension: {input_dim}")
    print(f"Output dimension: {output_dim}")

    model = PointMLP(input_dim, output_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(1000):
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

        print(f"Epoch {epoch}: Loss {loss.item():.4f}")

    # model rollout for 100 timesteps from the first frames

