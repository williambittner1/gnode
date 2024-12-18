import plotly.graph_objects as go
import wandb
import numpy as np
import torch
import os
from pointcloud import DynamicPointcloud
import open3d as o3d
import threading
import time
import cv2


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
        frames=frames#[0:]
    )
    
    # Update layout with both play and pause buttons
    fig.update_layout(
        scene=dict(
            dragmode=False, 
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
                    'args': [None, {'frame': {'duration': 10, 'redraw': True}, 'fromcurrent': True}]  # Set duration for 30 FPS
                },
                {
                    'label': 'Pause',
                    'method': 'animate',
                    'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate'}]
                },
                {
                    'label': 'Reset',
                    'method': 'animate',
                    'args': [[frames[0].name], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate'}]
                }
            ]
        }],
        # sliders=[{
        #     'currentvalue': {'prefix': 'Frame: '},
        #     'steps': [
        #         {
        #             'method': 'animate',
        #             'label': str(frame_num),
        #             'args': [[f"frame_{frame_num}"], {'frame': {'duration': 0, 'redraw': True}}]
        #         }
        #         for frame_num in frame_nums
        #     ]
        # }]
    )
    
    return fig


def compute_extrapolation_divergence(gt_dyn_pc, pred_dyn_pc, frame_nums):
    """Compute mean squared error between ground truth and predicted positions over time.
    
    Args:
        gt_dyn_pc: Ground truth DynamicPointcloud
        pred_dyn_pc: Predicted DynamicPointcloud
        frame_nums: List of frame numbers to evaluate
        
    Returns:
        errors: List of MSE values for each frame
    """
    errors = []
    
    for frame_num in frame_nums:
        if frame_num in gt_dyn_pc.frames and frame_num in pred_dyn_pc.frames:
            gt_pos = gt_dyn_pc.frames[frame_num].positions
            pred_pos = pred_dyn_pc.frames[frame_num].positions
            
            # Compute MSE for this frame
            frame_error = np.mean(np.square(gt_pos - pred_pos))
            errors.append(frame_error)
    
    return errors


def log_extrapolation_metrics(gt_dyn_pc, pred_dyn_pc, epoch):
    """
    Compute and return extrapolation metrics (without logging).
    
    Args:
        gt_dyn_pc: Ground truth DynamicPointcloud
        pred_dyn_pc: Predicted DynamicPointcloud
        epoch: Current training epoch
    Returns:
        dict: Dictionary containing the metrics
    """
    # Compute extrapolation divergence
    frame_nums = sorted(set(gt_dyn_pc.frames.keys()) & set(pred_dyn_pc.frames.keys()))
    divergence_errors = compute_extrapolation_divergence(gt_dyn_pc, pred_dyn_pc, frame_nums)
    
    # Return metrics dictionary without logging
    return {
        "extrapolation/mse_over_time": wandb.plot.line_series(
            xs=frame_nums,
            ys=[divergence_errors],
            keys=["MSE"],
            title=f"Extrapolation Divergence (Epoch {epoch})",
            xname="Frame"
        ),
        "extrapolation/mean_mse": np.mean(divergence_errors)
    }


def log_prediction_visualization(gt_dyn_pc, pred_dyn_pc, epoch):
    """
    Create and log visualization of ground truth vs prediction to wandb.
    
    Args:
        gt_dyn_pc: Ground truth DynamicPointcloud
        pred_dyn_pc: Predicted DynamicPointcloud
        epoch: Current training epoch
    """
    # Create visualization
    fig = create_plotly_figure(
        dyn_pcs_list=[gt_dyn_pc, pred_dyn_pc],
        dyn_pcs_names_list=["ground-truth", "prediction"]
    )
    
    # Convert to HTML and log
    html_str = fig.to_html(full_html=False, include_plotlyjs='cdn')
    wandb.log({
        "gt_vs_pred": wandb.Html(html_str, inject=False),
        "epoch": epoch
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


class DynamicPredictionViewer:
    PRED_GEOM_NAME = "Prediction"
    GT_GEOM_NAME = "Ground Truth"

    def __init__(self, gt_dyn_pc, pred_dyn_pc, update_delay=0.1):
        self.gt_dyn_pc = gt_dyn_pc
        self.pred_dyn_pc = pred_dyn_pc
        self.update_delay = update_delay
        self.is_done = False
        self.lock = threading.Lock()
        self.current_frame = 1
        
        # Get common frame numbers between gt and pred
        self.frame_nums = sorted(set(gt_dyn_pc.frames.keys()) & set(pred_dyn_pc.frames.keys()))

        # Initialize Open3D GUI
        self.app = o3d.visualization.gui.Application.instance
        self.window = self.app.create_window("Dynamic Point Cloud Prediction", width=1024, height=768)
        self.window.set_on_close(self.on_window_close)
        
        # Create layout
        em = self.window.theme.font_size
        margin = o3d.visualization.gui.Margins(em)
        
        # Create a layout
        layout = o3d.visualization.gui.Vert(0, margin)
        
        # Create SceneWidget for 3D visualization
        self.widget = o3d.visualization.gui.SceneWidget()
        self.widget.scene = o3d.visualization.rendering.Open3DScene(self.window.renderer)
        self.widget.scene.set_background([0.1, 0.1, 0.1, 1.0])
        layout.add_child(self.widget)

        # Create control panel
        control_layout = o3d.visualization.gui.Vert(0, margin)
        
        # Add slider for frame control
        # self.slider = o3d.visualization.gui.Slider()
        self.slider = o3d.visualization.gui.Slider(o3d.visualization.gui.Slider.INT)
        self.slider.set_limits(0, len(self.frame_nums) - 1)
        self.slider.set_on_value_changed(self.on_slider_changed)
        control_layout.add_child(self.slider)
        
        # Add play/pause button
        self.play_button = o3d.visualization.gui.Button("Play")
        self.play_button.set_on_clicked(self.on_play_button)
        control_layout.add_child(self.play_button)
        
        layout.add_child(control_layout)
        self.window.add_child(layout)

        # Initialize point clouds
        self.gt_pcd = o3d.geometry.PointCloud()
        self.pred_pcd = o3d.geometry.PointCloud()
        
        # Set materials
        self.gt_mat = o3d.visualization.rendering.MaterialRecord()
        self.gt_mat.shader = 'defaultUnlit'
        self.gt_mat.point_size = 3.0
        self.gt_mat.base_color = [0, 0, 1, 1.0]  # Blue for ground truth

        self.pred_mat = o3d.visualization.rendering.MaterialRecord()
        self.pred_mat.shader = 'defaultUnlit'
        self.pred_mat.point_size = 3.0
        self.pred_mat.base_color = [1, 0, 0, 1.0]  # Red for prediction

        # Add initial geometries
        self.update_point_clouds(self.frame_nums[0])
        
        # Setup camera
        bounds = self.widget.scene.bounding_box
        self.widget.setup_camera(60, bounds, bounds.get_center())

    def update_point_clouds(self, frame_num):
        # Update ground truth
        gt_points = self.gt_dyn_pc.frames[frame_num].positions
        self.gt_pcd.points = o3d.utility.Vector3dVector(gt_points)
        
        # Update prediction
        pred_points = self.pred_dyn_pc.frames[frame_num].positions
        self.pred_pcd.points = o3d.utility.Vector3dVector(pred_points)
        
        # Update scene
        self.widget.scene.remove_geometry(self.GT_GEOM_NAME)
        self.widget.scene.remove_geometry(self.PRED_GEOM_NAME)
        self.widget.scene.add_geometry(self.GT_GEOM_NAME, self.gt_pcd, self.gt_mat)
        self.widget.scene.add_geometry(self.PRED_GEOM_NAME, self.pred_pcd, self.pred_mat)

    def on_slider_changed(self, value):
        frame_idx = int(value)
        self.current_frame = self.frame_nums[frame_idx]
        self.update_point_clouds(self.current_frame)

    def on_play_button(self):
        if self.play_button.text == "Play":
            self.play_button.text = "Pause"
            threading.Thread(target=self.animation_thread).start()
        else:
            self.play_button.text = "Play"
            self.is_done = True

    def animation_thread(self):
        self.is_done = False
        while not self.is_done:
            time.sleep(self.update_delay)
            with self.lock:
                if self.is_done:
                    break
                # Update slider and frame
                current_idx = self.frame_nums.index(self.current_frame)
                next_idx = (current_idx + 1) % len(self.frame_nums)
                self.app.post_to_main_thread(
                    self.window, 
                    lambda: self.slider.set_value(next_idx)
                )

    def on_window_close(self):
        self.is_done = True
        return True
    
