import plotly.graph_objects as go
import wandb
import numpy as np
import torch
import os
from pointcloud import DynamicPointcloud

def create_plotly_figure(dyn_pcs_list, dyn_pcs_names_list, point_size=1, colors=None):
    """Create a figure comparing multiple dynamic pointclouds.
        
    Args:
        dyn_pcs_list: List of dynamic pointcloud objects to compare
        dyn_pcs_names_list: List of names for each pointcloud
        point_size: Size of points in visualization
        colors: Optional list of colors for each pointcloud. If None, uses default colors.
    Returns:
        fig: Plotly figure
    """
    frames = []

    # Set default colors if not provided
    if colors is None:
        colors = ['blue', 'red', 'green', 'purple', 'orange', 'cyan', 'magenta', 'yellow']
        # Repeat colors if more pointclouds than colors
        colors = colors * (len(dyn_pcs_list) // len(colors) + 1)
    
    # Get all frame numbers and sort them
    all_frames = [sorted(pc.frames.keys()) for pc in dyn_pcs_list]
    frame_nums = sorted(set.intersection(*[set(f) for f in all_frames]))
    
    # Create frames
    for frame_num in frame_nums:
        frame_data = []
        for i, dyn_pc in enumerate(dyn_pcs_list):
            pc = dyn_pc.frames[frame_num]
            trace = pc.to_plotly_trace(
                name=dyn_pcs_names_list[i],
                color=colors[i], 
                size=point_size
            )
            frame_data.append(trace)
            
        frame = go.Frame(
            data=frame_data,
            name=f"frame_{frame_num}"
        )
        frames.append(frame)
    
    # Create figure with the first frame
    fig = go.Figure(
        data=frames[0].data,
        frames=frames[1:]
    )
    
    # Update layout with both play and pause buttons
    fig.update_layout(
        scene=dict(
            aspectmode='cube',
            xaxis=dict(range=[-12, 12]),
            yaxis=dict(range=[-12, 12]),
            zaxis=dict(range=[-12, 12]),
        ),
        updatemenus=[{
            'type': 'buttons',
            'showactive': False,
            'buttons': [
                {
                    'label': 'Play',
                    'method': 'animate',
                    'args': [None, {'frame': {'duration': 33, 'redraw': True}, 'fromcurrent': True}]  # Set duration for 30 FPS
                },
                {
                    'label': 'Pause',
                    'method': 'animate',
                    'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate'}]
                }
            ]
        }],
        sliders=[{
            'currentvalue': {'prefix': 'Frame: '},
            'steps': [
                {
                    'method': 'animate',
                    'label': str(frame_num),
                    'args': [[f"frame_{frame_num}"], {'frame': {'duration': 0, 'redraw': True}}]
                }
                for frame_num in frame_nums
            ]
        }]
    )
    
    return fig


def log_visualization(model, test_dataset, device, config, epoch):
    model.eval()
    initial_X = test_dataset[1][0]  # dataset[1] gives (X_t, Y_t), [0] takes X_t
    test_sequence_path = os.path.join(test_dataset.split_dir, test_dataset.sequence_files[0])
    gt_dyn_pc = DynamicPointcloud()
    gt_dyn_pc.load_h5_sequence(test_sequence_path)
    with torch.no_grad():
        pred_dyn_pc = model.rollout(
            initial_X.to(device),
            rollout_length=config.rollout_length
        )
    fig = create_plotly_figure(
        dyn_pcs_list=[gt_dyn_pc, pred_dyn_pc],
        dyn_pcs_names_list=["ground-truth", "prediction"]
    )
    html_str = fig.to_html(full_html=False, include_plotlyjs='cdn')
    wandb.log({
        "gt_vs_pred": wandb.Html(html_str, inject=False)
    })
    print(f"Logged prediction visualization for epoch {epoch}")


def log_dataset_visualizations(dataset, num_sequences=5, prefix=""):
    """
    Log visualizations for multiple gt-sequences from a dataset (train or test).
    
    Args:
        dataset: Dataset containing sequences
        num_sequences: Number of sequences to visualize
        prefix: Prefix for wandb logging key (e.g., "train" or "test")
    """
    for i in range(min(num_sequences, len(dataset.sequence_files))):
        sequence_path = os.path.join(dataset.split_dir, dataset.sequence_files[i])
        dyn_pc = DynamicPointcloud()
        dyn_pc.load_h5_sequence(sequence_path)

        fig = create_plotly_figure(dyn_pc)
        html_str = fig.to_html(full_html=False, include_plotlyjs='cdn')
        wandb.log({
            f"{prefix}sequence_{i}": wandb.Html(html_str, inject=False)
        })