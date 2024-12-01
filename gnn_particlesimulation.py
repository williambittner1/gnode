# WORKING CODE WITH GNN INTEGRATION AND VISUALIZATION LOGGING

import matplotlib.pyplot as plt
from tqdm import tqdm  # Import tqdm for progress bar
import numpy as np
import plotly.graph_objs as go
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import MessagePassing, radius_graph
import wandb
import time
import colorsys

# ========================
# Initialize Weights & Biases
# ========================
wandb.init(project='multi-body-gnn', name='particle-dynamics-gnn')

# ========================
# Simulation Parameters
# ========================
NUM_PARTICLES = 6
INFLUENCE_RADIUS = 30.0  # Radius for connecting edges
G = 1.00  # Gravitational constant
TIME_STEP = 0.25  # Time step for the simulation
SIMULATION_STEPS = 100  # Number of simulation steps
VIEWPORT_WORLD_SIZE = 100
COLLISION_DISTANCE = 0.001  # Distance at which collisions occur
MAXIMUM_EDGE_WIDTH = 8
MAXIMUM_GRAVITATIONAL_FORCE = 10.0  # Maximum cap for gravitational force

# ========================
# Particle Class
# ========================
class Particle:
    def __init__(self, mass, position, velocity):
        self.mass = mass
        self.position = np.array(position, dtype=np.float64)
        self.velocity = np.array(velocity, dtype=np.float64)
        self.acceleration = np.zeros(3, dtype=np.float64)

    def update_acceleration(self, particles):
        force = np.zeros(3, dtype=np.float64)
        for other in particles:
            if other is self:
                continue
            r = other.position - self.position
            distance = np.linalg.norm(r)
            if 0.1 < distance < INFLUENCE_RADIUS:
                # Gravitational force calculation
                gravitational_force = G * self.mass * other.mass * r / (distance**3 + 1e-6)  # Added small epsilon

                # Cap the force by maximum gravitational force
                force_magnitude = np.linalg.norm(gravitational_force)
                if force_magnitude > MAXIMUM_GRAVITATIONAL_FORCE:
                    gravitational_force = (gravitational_force / force_magnitude) * MAXIMUM_GRAVITATIONAL_FORCE

                force += gravitational_force
        self.acceleration = force / self.mass

    def update_position_and_velocity(self, dt):
        self.velocity += self.acceleration * dt
        self.position += self.velocity * dt

# ========================
# Collision Handling Functions
# ========================
def handle_collisions(particles):
    """
    Detects and handles collisions between particles by updating their velocities
    based on elastic collision equations.

    Returns:
        colliding_pairs (list of tuples): List of particle index pairs that collided.
    """
    colliding_pairs = []
    for i in range(len(particles)):
        for j in range(i + 1, len(particles)):
            p1 = particles[i]
            p2 = particles[j]
            r = p1.position - p2.position
            distance = np.linalg.norm(r)
            if distance < COLLISION_DISTANCE:
                if distance == 0:
                    # Prevent division by zero by introducing a small random displacement
                    displacement = np.random.uniform(-1e-6, 1e-6, size=3)
                    p1.position += displacement
                    r = p1.position - p2.position
                    distance = np.linalg.norm(r)
                    if distance == 0:
                        continue  # Skip if still zero

                # Normalize the vector between particles
                n = r / distance
                # Relative velocity
                v_rel = p1.velocity - p2.velocity
                # Velocity component along the normal direction
                v_rel_dot_n = np.dot(v_rel, n)

                if v_rel_dot_n >= 0:
                    # Particles are moving apart; no collision
                    continue

                # Compute impulse scalar
                impulse = (2 * v_rel_dot_n) / (p1.mass + p2.mass)
                # Update velocities based on impulse
                p1.velocity -= (impulse * p2.mass) * n
                p2.velocity += (impulse * p1.mass) * n

                # Adjust positions to prevent overlap
                overlap = COLLISION_DISTANCE - distance
                p1.position += (overlap * p2.mass / (p1.mass + p2.mass)) * n
                p2.position -= (overlap * p1.mass / (p1.mass + p2.mass)) * n

                # Record the collision pair
                colliding_pairs.append( (i, j) )

    return colliding_pairs

def handle_boundary_collisions(particles, viewport_size):
    """
    Detects and handles collisions of particles with the simulation boundaries.
    Particles bounce off the walls elastically.
    """
    half_size = viewport_size / 2
    for particle in particles:
        for dim in range(3):  # Check x, y, z
            if particle.position[dim] <= -half_size:
                particle.position[dim] = -half_size
                if particle.velocity[dim] < 0:
                    particle.velocity[dim] *= -1  # Invert velocity
            elif particle.position[dim] >= half_size:
                particle.position[dim] = half_size
                if particle.velocity[dim] > 0:
                    particle.velocity[dim] *= -1  # Invert velocity

# ========================
# Edge Computation Function
# ========================
def compute_edges(particles, influence_radius):
    """
    Computes all unique pairs of particles within the influence radius and their gravitational forces.
    Returns:
        edges (list of tuples): List of particle index pairs that are connected.
        widths (list of floats): Corresponding widths for each edge based on gravitational force.
    """
    edges = []
    widths = []
    positions = [p.position for p in particles]
    for i, pos1 in enumerate(positions):
        for j, pos2 in enumerate(positions):
            if i < j and np.linalg.norm(pos1 - pos2) < influence_radius:
                r = pos2 - pos1
                distance = np.linalg.norm(r)
                if distance > 0:
                    # Calculate gravitational force
                    gravitational_force = G * particles[i].mass * particles[j].mass * r / (distance**3 + 1e-6)  # Added epsilon

                    force_magnitude = np.linalg.norm(gravitational_force)
                    # Normalize the force magnitude to a width between 0 and maximum_edge_width
                    normalized_width = min(max(force_magnitude / MAXIMUM_GRAVITATIONAL_FORCE * MAXIMUM_EDGE_WIDTH, 0), MAXIMUM_EDGE_WIDTH)
                    edges.append((i, j))
                    widths.append(normalized_width)
    return edges, widths

# ========================
# Particle Initialization Function
# ========================
def initialize_particles(num_particles, mass_range=(1, 10), position_range=(-10, 10), velocity_range=(-1, 1)):
    """
    Initializes particles with random masses, positions, and velocities.

    Args:
        num_particles (int): Number of particles to initialize.
        mass_range (tuple, optional): Range for particle masses. Defaults to (1, 10).
        position_range (tuple, optional): Range for particle positions. Defaults to (-10, 10).
        velocity_range (tuple, optional): Range for particle velocities. Defaults to (-1, 1).

    Returns:
        list: List of initialized particles.
    """
    particles = [
        Particle(
            mass=np.random.uniform(mass_range[0], mass_range[1]),
            position=np.random.uniform(position_range[0], position_range[1], size=3),
            velocity=np.random.uniform(velocity_range[0], velocity_range[1], size=3),
        )
        for _ in range(num_particles)
    ]
    return particles

# ========================
# GNN Classes
# ========================
class InteractionGNN(MessagePassing):
    def __init__(self, node_feature_dim=6, edge_feature_dim=3, hidden_dim=64):
        super(InteractionGNN, self).__init__(aggr='add')
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.mlp = nn.Sequential(
            nn.Linear(2 * node_feature_dim + edge_feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(node_feature_dim + hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 3)
        )

    def forward(self, x, edge_index, edge_attr):
        delta_x = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return delta_x

    def message(self, x_i, x_j, edge_attr):
        msg_input = torch.cat([x_i, x_j, edge_attr], dim=-1)
        msg = self.mlp(msg_input)
        return msg

    def update(self, aggr_out, x):
        update_input = torch.cat([x, aggr_out], dim=-1)
        delta_x = self.update_mlp(update_input)
        return delta_x

class ParticleNet(nn.Module):
    def __init__(self, node_feature_dim=6, edge_feature_dim=3, hidden_dim=64):
        super(ParticleNet, self).__init__()
        self.gnn = InteractionGNN(node_feature_dim, edge_feature_dim, hidden_dim)

    def forward(self, data):
        delta_x = self.gnn(data.x, data.edge_index, data.edge_attr)
        return delta_x

# ========================
# Generate Unique Colors for Visualization
# ========================
def generate_colors(num_colors):
    cmap = plt.cm.get_cmap("tab20", num_colors)  # Use a qualitative colormap
    return [cmap(i)[:3] for i in range(num_colors)]

# ========================
# Initialize Particles
# ========================
particles = initialize_particles(NUM_PARTICLES)

# ========================
# Visualization Data Storage
# ========================
frames = []

# Precompute half viewport size for visualization
half_viewport = VIEWPORT_WORLD_SIZE / 2

# Generate a list of unique colors in RGB format for edges
object_colors = generate_colors(NUM_PARTICLES)  # Assuming each particle is an object
object_hex_colors = [f"rgb({int(c[0]*255)}, {int(c[1]*255)}, {int(c[2]*255)})" for c in object_colors]

# ========================
# Define Test Initial Conditions
# ========================
# Set a fixed seed for reproducibility
np.random.seed(999)
test_particles = initialize_particles(NUM_PARTICLES, mass_range=(1, 10), position_range=(-10, 10), velocity_range=(-1, 1))
test_positions_initial = np.array([p.position for p in test_particles])
test_velocities_initial = np.array([p.velocity for p in test_particles])

# ========================
# Simulation Loop with Progress Bar
# ========================
for step in tqdm(range(SIMULATION_STEPS), desc="Processing Timesteps"):
    # Update accelerations based on gravitational forces
    for particle in particles:
        particle.update_acceleration(particles)

    # Update positions and velocities based on current accelerations
    for particle in particles:
        particle.update_position_and_velocity(TIME_STEP)

    # Handle collisions after updating positions and velocities
    colliding_pairs = handle_collisions(particles)

    # Handle boundary collisions to keep particles within the viewport
    handle_boundary_collisions(particles, VIEWPORT_WORLD_SIZE)

    # Extract positions and edges for visualization
    positions = np.array([p.position for p in particles])
    edges, widths = compute_edges(particles, INFLUENCE_RADIUS)

    # Separate edges into collision and normal edges
    # Convert colliding_pairs to a set of sorted tuples for efficient lookup
    colliding_set = set(tuple(sorted(pair)) for pair in colliding_pairs)

    normal_edges = []
    collision_edges = []

    for edge, width in zip(edges, widths):
        sorted_edge = tuple(sorted(edge))
        if sorted_edge in colliding_set:
            collision_edges.append(edge)
        else:
            # Optionally, differentiate edges within the same object
            # For simplicity, treating all non-colliding edges as normal
            normal_edges.append(edge)

    # Create particle trace
    particle_trace = go.Scatter3d(
        x=positions[:, 0],
        y=positions[:, 1],
        z=positions[:, 2],
        mode="markers",
        marker=dict(size=4, color="blue"),
        name="Particles   ",
    )

    # Create object-specific edge traces with unique colors
    object_edge_traces = []
    for idx, (i, j) in enumerate(edges):
        if idx >= MAXIMUM_EDGE_WIDTH:
            break  # Prevent excessive traces
        pos1 = positions[i]
        pos2 = positions[j]
        color = object_hex_colors[i]  # Assign color based on the first particle
        object_edge_traces.append(go.Scatter3d(
            x=[pos1[0], pos2[0], None],
            y=[pos1[1], pos2[1], None],
            z=[pos1[2], pos2[2], None],
            mode="lines",
            line=dict(width=widths[idx]*2, color=color),
            name=f"Edge {i}-{j}",
            opacity=0.8
        ))

    # Create normal edge trace (grey)
    edge_x_normal = []
    edge_y_normal = []
    edge_z_normal = []
    for edge, width in zip(normal_edges, widths):
        pos1 = positions[edge[0]]
        pos2 = positions[edge[1]]
        edge_x_normal += [pos1[0], pos2[0], None]
        edge_y_normal += [pos1[1], pos2[1], None]
        edge_z_normal += [pos1[2], pos2[2], None]

    edge_trace_normal = go.Scatter3d(
        x=edge_x_normal,
        y=edge_y_normal,
        z=edge_z_normal,
        mode="lines",
        line=dict(width=1, color="grey"),
        name="Normal Edges",
        opacity=0.3
    )

    # Create collision edge trace (red)
    edge_x_collision = []
    edge_y_collision = []
    edge_z_collision = []
    for edge in collision_edges:
        pos1 = positions[edge[0]]
        pos2 = positions[edge[1]]
        edge_x_collision += [pos1[0], pos2[0], None]
        edge_y_collision += [pos1[1], pos2[1], None]
        edge_z_collision += [pos1[2], pos2[2], None]

    edge_trace_collision = go.Scatter3d(
        x=edge_x_collision,
        y=edge_y_collision,
        z=edge_z_collision,
        mode="lines",
        line=dict(width=2, color="red"),
        name="Collision Edges",
        opacity=0.6
    )

    # Compile all traces for the current frame
    frame_data = [particle_trace] + object_edge_traces + [edge_trace_normal, edge_trace_collision]

    # Add frame data with updated annotations
    frames.append(go.Frame(
        data=frame_data,
        layout=go.Layout(
            annotations=[
                {
                    'text': f'Timestep: {step + 1} / {SIMULATION_STEPS}',
                    'x': 0.5,
                    'y': 1.05,
                    'xref': 'paper',
                    'yref': 'paper',
                    'showarrow': False,
                    'font': {'size': 16}
                }
            ]
        )
    ))

# ========================
# Plotly Figure Setup
# ========================
# Create initial figure layout
layout = go.Layout(
    scene=dict(
        xaxis=dict(range=[-half_viewport, half_viewport], autorange=False, title='X-axis'),
        yaxis=dict(range=[-half_viewport, half_viewport], autorange=False, title='Y-axis'),
        zaxis=dict(range=[-half_viewport, half_viewport], autorange=False, title='Z-axis'),
        aspectmode='cube'
    ),
    margin=dict(l=0, r=0, t=50, b=0),
    title="Dynamic Particle Simulation with Object-Specific Edges",
    updatemenus=[
        dict(
            type="buttons",
            showactive=False,
            buttons=[
                dict(
                    label="Play",
                    method="animate",
                    args=[
                        None,
                        dict(
                            frame=dict(duration=50, redraw=True),
                            transition=dict(duration=0),
                            fromcurrent=True,
                            mode="immediate"
                        ),
                    ],
                ),
                dict(
                    label="Pause",
                    method="animate",
                    args=[[None], dict(
                        mode="immediate",
                        frame=dict(duration=0, redraw=False),
                        transition=dict(duration=0)
                    )],
                )
            ],
        )
    ],
    legend=dict(
        orientation="h",  # Horizontal orientation
        yanchor="bottom",
        y=1.02,  # Position above the plot
        xanchor="right",
        x=1  # Align to the right
    )
)

# Combine initial data and frames into a figure
fig = go.Figure(
    data=frames[0].data,  # Initial frame's data
    layout=layout,
    frames=frames[1:],  # All frames except the first one
)

# Show animation
fig.show()

# ========================
# GNN Integration (One-Step Ahead Prediction)
# ========================
# Define GNN Classes (Already Defined Above)

# ========================
# Prepare Data for GNN Training
# ========================
# Simulation Parameters for GNN
NUM_PARTICLES_GNN = NUM_PARTICLES  # Ensure consistency
NUM_TIMESTEPS_GNN = 100
N_SYSTEMS = 1000  # Number of different sequences with varying initial conditions

# Initialize list to store Data objects
data_list = []

# Generate multiple sequences with different initial conditions
for system_idx in range(N_SYSTEMS):
    # Initialize particles with random positions and velocities
    system_particles = initialize_particles(NUM_PARTICLES_GNN, mass_range=(1, 10), position_range=(-10, 10), velocity_range=(-1, 1))

    # Simulate the system for NUM_TIMESTEPS_GNN
    positions = []
    velocities = []
    for step in range(NUM_TIMESTEPS_GNN):
        # Record current positions and velocities
        positions.append(np.array([p.position for p in system_particles]))
        velocities.append(np.array([p.velocity for p in system_particles]))

        # Update accelerations
        for particle in system_particles:
            particle.update_acceleration(system_particles)

        # Update positions and velocities
        for particle in system_particles:
            particle.update_position_and_velocity(TIME_STEP)

        # Handle collisions and boundary conditions
        handle_collisions(system_particles)
        handle_boundary_collisions(system_particles, VIEWPORT_WORLD_SIZE)

    # Prepare Data objects for GNN (one-step ahead)
    for t in range(NUM_TIMESTEPS_GNN - 1):
        # Current state
        positions_t = torch.tensor(positions[t], dtype=torch.float32)
        velocities_t = torch.tensor(velocities[t], dtype=torch.float32)
        x = torch.cat([positions_t, velocities_t], dim=-1)  # Shape: [NUM_PARTICLES_GNN, 6]

        # Next state (target)
        positions_t1 = torch.tensor(positions[t + 1], dtype=torch.float32)
        y = positions_t1  # Predict next positions

        # Create edge index using radius_graph
        pos_tensor = torch.tensor(positions[t], dtype=torch.float32)
        edge_index = radius_graph(pos_tensor, r=INFLUENCE_RADIUS, loop=False)

        # Compute edge attributes (displacement vectors)
        row, col = edge_index
        edge_attr = pos_tensor[col] - pos_tensor[row]  # Shape: [num_edges, 3]

        # Create Data object
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        data.timestep = t
        data.system_idx = system_idx  # Optionally, add system index
        data_list.append(data)

# ========================
# Initialize GNN, Loss, Optimizer
# ========================
# Initialize GNN
model = ParticleNet(node_feature_dim=6, edge_feature_dim=3, hidden_dim=64)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

# Define Loss Function and Optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.9)

# ========================
# Data Loader
# ========================
batch_size = 256
loader = DataLoader(data_list, batch_size=batch_size, shuffle=True)

# ========================
# Define Functions for Rollout and Visualization
# ========================
def simulate_ground_truth(initial_positions, initial_velocities, n_steps=200, dt=TIME_STEP):
    """
    Simulates the ground truth dynamics for n_steps starting from initial_positions and initial_velocities.

    Args:
        initial_positions (np.ndarray): Initial positions of particles, shape [num_particles, 3].
        initial_velocities (np.ndarray): Initial velocities of particles, shape [num_particles, 3].
        n_steps (int): Number of simulation steps.
        dt (float): Time step.

    Returns:
        positions (np.ndarray): Positions over time, shape [n_steps, num_particles, 3].
        velocities (np.ndarray): Velocities over time, shape [n_steps, num_particles, 3].
    """
    # Initialize particles
    particles = []
    for i in range(initial_positions.shape[0]):
        p = Particle(
            mass=np.random.uniform(1, 10),  # Or use fixed masses if desired
            position=initial_positions[i],
            velocity=initial_velocities[i]
        )
        particles.append(p)

    positions = []
    velocities = []

    for step in range(n_steps):
        positions.append(np.array([p.position.copy() for p in particles]))
        velocities.append(np.array([p.velocity.copy() for p in particles]))

        # Update accelerations
        for p in particles:
            p.update_acceleration(particles)

        # Update positions and velocities
        for p in particles:
            p.update_position_and_velocity(dt)

        # Handle collisions
        handle_collisions(particles)

        # Handle boundary collisions
        handle_boundary_collisions(particles, VIEWPORT_WORLD_SIZE)

    positions = np.array(positions)  # Shape: [n_steps, num_particles, 3]
    velocities = np.array(velocities)  # Shape: [n_steps, num_particles, 3]
    return positions, velocities

def simulate_gnn_predictions(model, initial_positions, initial_velocities, n_steps=200, dt=TIME_STEP, influence_radius=INFLUENCE_RADIUS):
    """
    Simulates dynamics using the trained GNN for n_steps starting from initial_positions and initial_velocities.

    Args:
        model (torch.nn.Module): Trained GNN model.
        initial_positions (np.ndarray): Initial positions of particles, shape [num_particles, 3].
        initial_velocities (np.ndarray): Initial velocities of particles, shape [num_particles, 3].
        n_steps (int): Number of prediction steps.
        dt (float): Time step.
        influence_radius (float): Interaction radius for edge creation.

    Returns:
        positions_pred (np.ndarray): Predicted positions over time, shape [n_steps, num_particles, 3].
    """
    model.eval()
    device = next(model.parameters()).device

    # Initialize positions and velocities
    positions_pred = []
    current_positions = initial_positions.copy()
    current_velocities = initial_velocities.copy()

    for step in range(n_steps):
        positions_pred.append(current_positions.copy())

        # Prepare input tensor
        x = torch.tensor(np.hstack([current_positions, current_velocities]), dtype=torch.float32).to(device)  # Shape: [num_particles, 6]

        # Create edge index using radius_graph
        pos_tensor = torch.tensor(current_positions, dtype=torch.float32).to(device)
        edge_index = radius_graph(pos_tensor, r=influence_radius, loop=False)

        # Compute edge attributes (displacement vectors)
        row, col = edge_index
        edge_attr = pos_tensor[col] - pos_tensor[row]  # Shape: [num_edges, 3]

        # Create Data object
        data_input = Data(x=x, edge_index=edge_index, edge_attr=edge_attr).to(device)

        # Predict delta_x
        with torch.no_grad():
            delta_x = model(data_input).cpu().numpy()  # Shape: [num_particles, 3]

        # Update positions and velocities
        current_positions += delta_x * dt
        current_velocities = (current_positions - positions_pred[-1]) / dt  # Estimate new velocities

        # Handle collisions and boundary conditions
        # Create Particle instances temporarily for collision handling
        temp_particles = []
        for i in range(current_positions.shape[0]):
            p = Particle(
                mass=1.0,  # Assuming mass=1 for simplicity
                position=current_positions[i],
                velocity=current_velocities[i]
            )
            temp_particles.append(p)

        handle_collisions(temp_particles)
        handle_boundary_collisions(temp_particles, VIEWPORT_WORLD_SIZE)

        # Update positions and velocities after handling
        current_positions = np.array([p.position for p in temp_particles])
        current_velocities = np.array([p.velocity for p in temp_particles])

    positions_pred = np.array(positions_pred)  # Shape: [n_steps, num_particles, 3]
    return positions_pred


def create_comparison_animation_fixed_axes(positions_gt, positions_pred, n_steps=200, viewport_size=100):
    """
    Creates a Plotly animation comparing ground truth and GNN-predicted trajectories
    with fixed axes ranges and equal scaling.

    Args:
        positions_gt (np.ndarray): Ground truth positions over time, shape [n_steps, num_particles, 3].
        positions_pred (np.ndarray): Predicted positions over time, shape [n_steps, num_particles, 3].
        n_steps (int): Number of steps in the rollout.
        viewport_size (float): Size of the viewport for axis ranges.

    Returns:
        fig (plotly.graph_objs.Figure): Plotly figure with animation.
    """
    num_particles = positions_gt.shape[1]

    # Define colors for particles
    colors_gt = ['blue'] * num_particles
    colors_pred = ['orange'] * num_particles

    # Initialize figure
    fig = go.Figure()

    # Initial data
    for i in range(num_particles):
        fig.add_trace(go.Scatter3d(
            x=[positions_gt[0, i, 0]],
            y=[positions_gt[0, i, 1]],
            z=[positions_gt[0, i, 2]],
            mode='lines',
            line=dict(color=colors_gt[i], width=2),
            name=f'GT Particle {i+1}',
            showlegend=False
        ))
        fig.add_trace(go.Scatter3d(
            x=[positions_pred[0, i, 0]],
            y=[positions_pred[0, i, 1]],
            z=[positions_pred[0, i, 2]],
            mode='lines',
            line=dict(color=colors_pred[i], width=2),
            name=f'Pred Particle {i+1}',
            showlegend=False
        ))

    # Create frames
    frames_plotly = []
    for step in range(n_steps):
        frame_data = []
        for i in range(num_particles):
            # Ground truth
            frame_data.append(go.Scatter3d(
                x=positions_gt[:step+1, i, 0],
                y=positions_gt[:step+1, i, 1],
                z=positions_gt[:step+1, i, 2],
                mode='lines',
                line=dict(color=colors_gt[i], width=2),
                showlegend=False
            ))
            # GNN Predictions
            frame_data.append(go.Scatter3d(
                x=positions_pred[:step+1, i, 0],
                y=positions_pred[:step+1, i, 1],
                z=positions_pred[:step+1, i, 2],
                mode='lines',
                line=dict(color=colors_pred[i], width=2),
                showlegend=False
            ))
        frames_plotly.append(go.Frame(data=frame_data, name=f'Frame {step+1}'))

    # Assign frames
    fig.frames = frames_plotly

    # Set layout with fixed axes
    half_viewport = viewport_size / 2
    fig.update_layout(
        title="Ground Truth vs GNN Predicted Trajectories",
        scene=dict(
            xaxis=dict(range=[-half_viewport, half_viewport], autorange=False, title='X-axis'),
            yaxis=dict(range=[-half_viewport, half_viewport], autorange=False, title='Y-axis'),
            zaxis=dict(range=[-half_viewport, half_viewport], autorange=False, title='Z-axis'),
            aspectmode='cube'
        ),
        margin=dict(l=0, r=0, t=50, b=0),
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=50, redraw=True),
                                transition=dict(duration=0),
                                fromcurrent=True,
                                mode="immediate"
                            ),
                        ],
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[[None], dict(
                            mode="immediate",
                            frame=dict(duration=0, redraw=False),
                            transition=dict(duration=0)
                        )],
                    )
                ],
            )
        ],
        legend=dict(
            orientation="h",  # Horizontal orientation
            yanchor="bottom",
            y=1.02,  # Position above the plot
            xanchor="right",
            x=1  # Align to the right
        )
    )

    return fig

def create_comparison_animation(positions_gt, positions_pred, n_steps=200, name='comparison'):
    """
    Creates a Plotly animation comparing ground truth and GNN-predicted trajectories.
    Ground truth in blue and GNN predictions in orange.

    Args:
        positions_gt (np.ndarray): Ground truth positions over time, shape [n_steps, num_particles, 3].
        positions_pred (np.ndarray): Predicted positions over time, shape [n_steps, num_particles, 3].
        n_steps (int): Number of steps in the rollout.
        name (str): Name identifier for the animation.

    Returns:
        fig (plotly.graph_objs.Figure): Plotly figure with animation.
    """
    num_particles = positions_gt.shape[1]
    fig = go.Figure()

    # Define colors for particles
    colors_gt = ['blue'] * num_particles
    colors_pred = ['orange'] * num_particles

    # Initialize traces for ground truth and predictions
    for i in range(num_particles):
        # Ground Truth
        fig.add_trace(go.Scatter3d(
            x=[positions_gt[0, i, 0]],
            y=[positions_gt[0, i, 1]],
            z=[positions_gt[0, i, 2]],
            mode='lines',
            line=dict(color=colors_gt[i], width=2),
            name=f'GT Particle {i+1}',
            showlegend=False
        ))
        # GNN Predictions
        fig.add_trace(go.Scatter3d(
            x=[positions_pred[0, i, 0]],
            y=[positions_pred[0, i, 1]],
            z=[positions_pred[0, i, 2]],
            mode='lines',
            line=dict(color=colors_pred[i], width=2),
            name=f'Pred Particle {i+1}',
            showlegend=False
        ))

    # Create frames
    frames_plotly = []
    for step in range(n_steps):
        frame_data = []
        for i in range(num_particles):
            # Ground truth
            frame_data.append(go.Scatter3d(
                x=positions_gt[:step+1, i, 0],
                y=positions_gt[:step+1, i, 1],
                z=positions_gt[:step+1, i, 2],
                mode='lines',
                line=dict(color=colors_gt[i], width=2),
                showlegend=False
            ))
            # GNN Predictions
            frame_data.append(go.Scatter3d(
                x=positions_pred[:step+1, i, 0],
                y=positions_pred[:step+1, i, 1],
                z=positions_pred[:step+1, i, 2],
                mode='lines',
                line=dict(color=colors_pred[i], width=2),
                showlegend=False
            ))
        frames_plotly.append(go.Frame(data=frame_data, name=f'Frame {step+1}'))

    # Assign frames
    fig.frames = frames_plotly

    # Set layout
    fig.update_layout(
        title="Ground Truth vs GNN Predicted Trajectories",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode='cube'
        ),
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                buttons=[
                    dict(label="Play",
                         method="animate",
                         args=[
                             None,
                             dict(frame=dict(duration=50, redraw=True),
                                  transition=dict(duration=0),
                                  fromcurrent=True,
                                  mode='immediate')
                         ]),
                    dict(label="Pause",
                         method="animate",
                         args=[
                             [None],
                             dict(frame=dict(duration=0, redraw=False),
                                  transition=dict(duration=0),
                                  mode='immediate')
                         ])
                ]
            )
        ],
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    return fig



# ========================
# Training Loop with Visualization Logging
# ========================
num_epochs = 15000  # Adjust as needed

model.train()
progress_bar = tqdm(range(num_epochs), desc="Training Progress", unit="epoch")

for epoch in progress_bar:
    epoch_start_time = time.time()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        delta_x_pred = model(data)
        positions_t = data.x[:, :3]
        positions_pred = positions_t + delta_x_pred  # Predict next positions
        positions_true = data.y
        loss = criterion(positions_pred, positions_true)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]

    # Update progress bar
    progress_bar.set_description(
        f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(loader):.6f}, LR: {current_lr:.6f}"
    )

    # Visualization and Logging every 10 epochs
    if (epoch + 1) % 10 == 0:
        # Simulate ground truth
        positions_gt, velocities_gt = simulate_ground_truth(
            test_positions_initial, test_velocities_initial, n_steps=200, dt=TIME_STEP
        )

        # Simulate GNN predictions
        positions_pred = simulate_gnn_predictions(
            model,
            test_positions_initial,
            test_velocities_initial,
            n_steps=200,
            dt=TIME_STEP,
            influence_radius=INFLUENCE_RADIUS,
        )

        # Create animation
        fig_comparison = create_comparison_animation_fixed_axes(positions_gt, positions_pred, n_steps=200)

        # Export Plotly animation to HTML
        html_str = fig_comparison.to_html(full_html=False)

        # Log the HTML animation to wandb
        try:
            wandb.log(
                {f"3D Trajectories at Epoch": wandb.Html(html_str, inject=False)},
                step=epoch + 1,
            )
        except Exception as e:
            print(f"Error logging to wandb: {e}")
            print("Skipping figure logging for this epoch.")

    # Log training metrics to wandb
    wandb.log(
        {
            "epoch": epoch + 1,
            "loss": total_loss / len(loader),
            "learning_rate": current_lr,
        },
        step=epoch + 1,
    )

    # Optionally, add training speed metrics
    epoch_duration = time.time() - epoch_start_time
    epochs_per_second = 1 / epoch_duration
    wandb.log({"epochs_per_second": epochs_per_second}, step=epoch + 1)

# ========================
# Finish Weights & Biases Run
# ========================
wandb.finish()
