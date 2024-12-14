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
                    'args': [None, {'frame': {'duration': 50, 'redraw': True}, 'fromcurrent': True}]
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


def log_pointcloud_sequence(dyn_pcs, log_frequency=10, name_prefix="pointcloud"):
    """Log multiple dynamic pointclouds to wandb using Object3D.
    
    Args:
        dyn_pcs: List of DynamicPointcloud objects to compare
        log_frequency: Log every nth frame
        name_prefix: Prefix for the logged pointcloud names
    """
    # Get all frame numbers and sort them
    all_frames = [sorted(pc.frames.keys()) for pc in dyn_pcs]
    frame_nums = sorted(set.intersection(*[set(f) for f in all_frames]))
    
    # Default colors for different pointclouds
    colors = [(0, 0, 255), (255, 0, 0), (0, 255, 0), (255, 0, 255)]
    
    for frame_num in frame_nums[::log_frequency]:
        frame_data = {}
        
        for i, dyn_pc in enumerate(dyn_pcs):
            pc = dyn_pc.frames[frame_num]
            # Ensure points are numpy array
            points = np.array(pc.positions, dtype=np.float32)
            if points is None or len(points) == 0:
                continue
                
            # Create colored points array
            color = np.tile(colors[i % len(colors)], (len(points), 1)).astype(np.uint8)
            
            # Create Object3D data
            obj3d_data = {
                "type": "lidar/beta",
                "points": points,
                "colors": color,
                "boxes": None
            }
            
            try:
                frame_data[f"{name_prefix}_{i}"] = wandb.Object3D(obj3d_data)
            except Exception as e:
                print(f"Error creating Object3D for frame {frame_num}, pointcloud {i}: {e}")
                print(f"Points shape: {points.shape}, Colors shape: {color.shape}")
                continue
        
        # Add frame metadata
        frame_data["frame"] = frame_num
        
        # Only log if we have valid data
        if len(frame_data) > 1:  # More than just the frame number
            wandb.log(frame_data)
