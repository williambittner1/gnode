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
# Configuration and Parameters
# ========================
class Config:
    # Simulation Parameters
    NUM_PARTICLES = 3
    INFLUENCE_RADIUS = 30.0  # Radius for connecting edges
    G = 1.00  # Gravitational constant
    TIME_STEP = 0.25  # Time step for the simulation
    SIMULATION_STEPS = 100  # Number of simulation steps for final visualization/evaluation
    VIEWPORT_WORLD_SIZE = 100
    COLLISION_DISTANCE = 0.001  # Distance at which collisions occur
    MAXIMUM_EDGE_WIDTH = 8 # Maximum edge line width in visualizationwhen the force is at its maximum
    MAXIMUM_GRAVITATIONAL_FORCE = 10.0  # Maximum cap for gravitational force

    # GNN Parameters
    NODE_FEATURE_DIM = 6
    EDGE_FEATURE_DIM = 3
    HIDDEN_DIM = 64

    # Training Parameters
    NUM_EPOCHS = 15000
    BATCH_SIZE = 512
    LEARNING_RATE = 1e-3
    NUM_SYSTEMS = 1000
    NUM_TIMESTEPS_GNN = 500 # Number of timesteps each system is simulated for training

    # Visualization Parameters
    NUM_ROLLOUT_STEPS = 500 # Number of timesteps each system is simulated for visualization at wandb
    WANDB_VISUALIZATION_FREQUENCY = 100 # How often to log to wandb

    # Reproducibility
    RANDOM_SEED = 999


# ========================
# Particle Class
# ========================
class Particle:
    def __init__(self, mass, position, velocity):
        self.mass = mass
        self.position = np.array(position, dtype=np.float64)
        self.velocity = np.array(velocity, dtype=np.float64)
        self.acceleration = np.zeros(3, dtype=np.float64)

    def update_acceleration(self, particles, config: Config):
        force = np.zeros(3, dtype=np.float64)
        for other in particles:
            if other is self:
                continue
            r = other.position - self.position
            distance = np.linalg.norm(r)
            if 0.1 < distance < config.INFLUENCE_RADIUS:
                # Gravitational force calculation
                gravitational_force = config.G * self.mass * other.mass * r / (distance**3 + 1e-6)  # Added small epsilon

                # Cap the force by maximum gravitational force
                force_magnitude = np.linalg.norm(gravitational_force)
                if force_magnitude > config.MAXIMUM_GRAVITATIONAL_FORCE:
                    gravitational_force = (gravitational_force / force_magnitude) * config.MAXIMUM_GRAVITATIONAL_FORCE

                force += gravitational_force
        self.acceleration = force / self.mass

    def update_position_and_velocity(self, dt):
        self.velocity += self.acceleration * dt
        self.position += self.velocity * dt


# ========================
# Simulation Class
# ========================
class Simulation:
    def __init__(self, config: Config):
        self.config = config
        self.particles = self.initialize_particles()
        self.half_viewport = self.config.VIEWPORT_WORLD_SIZE / 2
        self.object_colors = self.generate_colors(self.config.NUM_PARTICLES)
        self.object_hex_colors = [f"rgb({int(c[0]*255)}, {int(c[1]*255)}, {int(c[2]*255)})" for c in self.object_colors]

    def initialize_particles(self, mass_range=(1, 1), position_range=(-10, 10), velocity_range=(-1, 1)):
        particles = [
            Particle(
                mass=np.random.uniform(mass_range[0], mass_range[1]),
                position=np.random.uniform(position_range[0], position_range[1], size=3),
                velocity=np.random.uniform(velocity_range[0], velocity_range[1], size=3),
            )
            for _ in range(self.config.NUM_PARTICLES)
        ]
        return particles

    def generate_colors(self, num_colors):
        cmap = plt.cm.get_cmap("tab20", num_colors)  # Use a qualitative colormap
        return [cmap(i)[:3] for i in range(num_colors)]

    def update(self):
        # Update accelerations based on gravitational forces
        for particle in self.particles:
            particle.update_acceleration(self.particles, self.config)

        # Update positions and velocities based on current accelerations
        for particle in self.particles:
            particle.update_position_and_velocity(self.config.TIME_STEP)

        # Handle collisions after updating positions and velocities
        colliding_pairs = self.handle_collisions()

        # Handle boundary collisions to keep particles within the viewport
        self.handle_boundary_collisions()

        # Extract positions and edges for visualization
        positions = np.array([p.position for p in self.particles])
        edges, widths = self.compute_edges()

        return positions, edges, widths, colliding_pairs

    def handle_collisions(self):
        colliding_pairs = []
        for i in range(len(self.particles)):
            for j in range(i + 1, len(self.particles)):
                p1 = self.particles[i]
                p2 = self.particles[j]
                r = p1.position - p2.position
                distance = np.linalg.norm(r)
                if distance < self.config.COLLISION_DISTANCE:
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
                    overlap = self.config.COLLISION_DISTANCE - distance
                    p1.position += (overlap * p2.mass / (p1.mass + p2.mass)) * n
                    p2.position -= (overlap * p1.mass / (p1.mass + p2.mass)) * n

                    # Record the collision pair
                    colliding_pairs.append((i, j))

        return colliding_pairs

    def handle_boundary_collisions(self):
        half_size = self.half_viewport
        for particle in self.particles:
            for dim in range(3):  # Check x, y, z
                if particle.position[dim] <= -half_size:
                    particle.position[dim] = -half_size
                    if particle.velocity[dim] < 0:
                        particle.velocity[dim] *= -1  # Invert velocity
                elif particle.position[dim] >= half_size:
                    particle.position[dim] = half_size
                    if particle.velocity[dim] > 0:
                        particle.velocity[dim] *= -1  # Invert velocity

    def compute_edges(self):
        edges = []
        widths = []
        positions = [p.position for p in self.particles]
        for i, pos1 in enumerate(positions):
            for j, pos2 in enumerate(positions):
                if i < j and np.linalg.norm(pos1 - pos2) < self.config.INFLUENCE_RADIUS:
                    r = pos2 - pos1
                    distance = np.linalg.norm(r)
                    if distance > 0:
                        # Calculate gravitational force
                        gravitational_force = self.config.G * self.particles[i].mass * self.particles[j].mass * r / (distance**3 + 1e-6)  # Added epsilon

                        force_magnitude = np.linalg.norm(gravitational_force)
                        # Normalize the force magnitude to a width between 0 and maximum_edge_width
                        normalized_width = min(max(force_magnitude / self.config.MAXIMUM_GRAVITATIONAL_FORCE * self.config.MAXIMUM_EDGE_WIDTH, 0), self.config.MAXIMUM_EDGE_WIDTH)
                        edges.append((i, j))
                        widths.append(normalized_width)
        return edges, widths

    def reset(self):
        self.particles = self.initialize_particles()


# ========================
# GNN Classes
# ========================
class InteractionGNN(MessagePassing):
    def __init__(self, node_feature_dim, edge_feature_dim, hidden_dim):
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
        # Ensure all tensors are on the same device
        device = x_i.device
        msg_input = torch.cat([x_i, x_j, edge_attr], dim=-1).to(device)
        # print(f"msg_input device: {msg_input.device}")  # Debugging
        msg = self.mlp(msg_input)
        return msg

    def update(self, aggr_out, x):
        # Ensure all tensors are on the same device
        device = x.device
        update_input = torch.cat([x, aggr_out], dim=-1).to(device)
        # print(f"update_input device: {update_input.device}")  # Debugging
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
# Visualizer Class
# ========================
class Visualizer:
    def __init__(self, config: Config, simulation: Simulation):
        self.config = config
        self.simulation = simulation
        self.frames = []
        self.object_colors = simulation.object_colors
        self.object_hex_colors = simulation.object_hex_colors

    def generate_frame(self, positions, edges, widths, colliding_pairs, step):
        # Separate edges into collision and normal edges
        colliding_set = set(tuple(sorted(pair)) for pair in colliding_pairs)
        normal_edges = []
        collision_edges = []

        for edge, width in zip(edges, widths):
            sorted_edge = tuple(sorted(edge))
            if sorted_edge in colliding_set:
                collision_edges.append(edge)
            else:
                normal_edges.append(edge)

        # Create particle trace
        particle_trace = go.Scatter3d(
            x=positions[:, 0],
            y=positions[:, 1],
            z=positions[:, 2],
            mode="markers",
            marker=dict(size=2, color="blue"),
            name="Particles",
        )

        # Create object-specific edge traces with unique colors
        object_edge_traces = []
        for idx, (i, j) in enumerate(edges):
            if idx >= self.config.MAXIMUM_EDGE_WIDTH:
                break  # Prevent excessive traces
            pos1 = positions[i]
            pos2 = positions[j]
            color = self.object_hex_colors[i]  # Assign color based on the first particle
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
        frame = go.Frame(
            data=frame_data,
            layout=go.Layout(
                annotations=[
                    {
                        'text': f'Timestep: {step + 1} / {self.config.SIMULATION_STEPS}',
                        'x': 0.5,
                        'y': 1.05,
                        'xref': 'paper',
                        'yref': 'paper',
                        'showarrow': False,
                        'font': {'size': 16}
                    }
                ]
            )
        )

        self.frames.append(frame)

    def create_animation(self):
        # Create initial figure layout
        layout = go.Layout(
            scene=dict(
                xaxis=dict(range=[-self.simulation.half_viewport, self.simulation.half_viewport], autorange=False, title='X-axis'),
                yaxis=dict(range=[-self.simulation.half_viewport, self.simulation.half_viewport], autorange=False, title='Y-axis'),
                zaxis=dict(range=[-self.simulation.half_viewport, self.simulation.half_viewport], autorange=False, title='Z-axis'),
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
        if self.frames:
            fig = go.Figure(
                data=self.frames[0].data,  # Initial frame's data
                layout=layout,
                frames=self.frames[1:],  # All frames except the first one
            )
            fig.show()

    def reset(self):
        self.frames = []


# ========================
# GNN Model Wrapper
# ========================
class GNNModelWrapper:
    def __init__(self, config: Config, device: str):
        self.config = config
        self.device = device
        self.model = ParticleNet(
            node_feature_dim=self.config.NODE_FEATURE_DIM,
            edge_feature_dim=self.config.EDGE_FEATURE_DIM,
            hidden_dim=self.config.HIDDEN_DIM
        ).to(self.device)

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.LEARNING_RATE)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=500, gamma=0.9)

    def train_step(self, batch):
        self.model.train()
        self.optimizer.zero_grad()
        delta_x_pred = self.model(batch)
        positions_t = batch.x[:, :3]
        positions_pred = positions_t + delta_x_pred  # Predict next positions
        positions_true = batch.y
        loss = self.criterion(positions_pred, positions_true)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def evaluate(self, data_loader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in data_loader:
                batch = batch.to(self.device)
                delta_x_pred = self.model(batch)
                positions_t = batch.x[:, :3]
                positions_pred = positions_t + delta_x_pred
                positions_true = batch.y
                loss = self.criterion(positions_pred, positions_true)
                total_loss += loss.item()
        return total_loss / len(data_loader)

    def step_scheduler(self):
        self.scheduler.step()

    def get_learning_rate(self):
        return self.scheduler.get_last_lr()[0]


# ========================
# Trainer Class
# ========================
class Trainer:
    def __init__(self, config: Config):
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        torch.manual_seed(self.config.RANDOM_SEED)
        np.random.seed(self.config.RANDOM_SEED)
        self.simulation = Simulation(config)
        self.visualizer = Visualizer(config, self.simulation)
        self.model_wrapper = GNNModelWrapper(config, self.device)
        self.data_loader = self.prepare_data()
        self.initialize_wandb()

        # Define test initial conditions
        self.test_particles = self.simulation.initialize_particles(
            mass_range=(1, 1),
            position_range=(-10, 10),
            velocity_range=(-1, 1)
        )
        self.test_positions_initial = np.array([p.position for p in self.test_particles])
        self.test_velocities_initial = np.array([p.velocity for p in self.test_particles])

    def initialize_wandb(self):
        wandb.init(project='gnode_test'),
        dir='/work/williamb/gaussian_ode'

    def prepare_data(self):
        data_list = []
        for system_idx in range(self.config.NUM_SYSTEMS):
            # Initialize particles with random positions and velocities
            system_particles = self.simulation.initialize_particles(
                mass_range=(1, 1),
                position_range=(-10, 10),
                velocity_range=(-1, 1)
            )

            # Simulate the system for NUM_TIMESTEPS_GNN
            positions = []
            velocities = []
            for step in range(self.config.NUM_TIMESTEPS_GNN):
                # Record current positions and velocities
                positions.append(np.array([p.position.copy() for p in system_particles]))
                velocities.append(np.array([p.velocity.copy() for p in system_particles]))

                # Update accelerations
                for particle in system_particles:
                    particle.update_acceleration(system_particles, self.config)

                # Update positions and velocities
                for particle in system_particles:
                    particle.update_position_and_velocity(self.config.TIME_STEP)

                # Handle collisions and boundary conditions
                self.simulation.handle_collisions()
                self.simulation.handle_boundary_collisions()

            # Prepare Data objects for GNN (one-step ahead)
            for t in range(self.config.NUM_TIMESTEPS_GNN - 1):
                # Current state
                positions_t = torch.tensor(positions[t], dtype=torch.float32)
                velocities_t = torch.tensor(velocities[t], dtype=torch.float32)
                x = torch.cat([positions_t, velocities_t], dim=-1)  # Shape: [NUM_PARTICLES_GNN, 6]

                # Next state (target)
                positions_t1 = torch.tensor(positions[t + 1], dtype=torch.float32)
                y = positions_t1  # Predict next positions

                # Create edge index using radius_graph
                pos_tensor = torch.tensor(positions[t], dtype=torch.float32)
                edge_index = radius_graph(pos_tensor, r=self.config.INFLUENCE_RADIUS, loop=False)

                # Compute edge attributes (displacement vectors)
                row, col = edge_index
                edge_attr = pos_tensor[col] - pos_tensor[row]  # Shape: [num_edges, 3]

                # Create Data object and move to device immediately
                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y).to(self.device)
                data.timestep = t
                data.system_idx = system_idx  # Optionally, add system index
                data_list.append(data)

        # Create DataLoader
        loader = DataLoader(data_list, batch_size=self.config.BATCH_SIZE, shuffle=True)
        return loader

    def simulate_ground_truth(self, initial_positions, initial_velocities, n_steps=None):
        """
        Simulates the ground truth dynamics for n_steps starting from initial_positions and initial_velocities.

        Args:
            initial_positions (np.ndarray): Initial positions of particles, shape [num_particles, 3].
            initial_velocities (np.ndarray): Initial velocities of particles, shape [num_particles, 3].
            n_steps (int, optional): Number of simulation steps. Defaults to self.config.NUM_ROLLOUT_STEPS.

        Returns:
            positions (np.ndarray): Positions over time, shape [n_steps, num_particles, 3].
            velocities (np.ndarray): Velocities over time, shape [n_steps, num_particles, 3].
        """
        if n_steps is None:
            n_steps = self.config.NUM_ROLLOUT_STEPS

        # Initialize particles
        particles = []
        for i in range(initial_positions.shape[0]):
            p = Particle(
                mass=np.random.uniform(1, 1),  # Or use fixed masses if desired
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
                p.update_acceleration(particles, self.config)

            # Update positions and velocities
            for p in particles:
                p.update_position_and_velocity(self.config.TIME_STEP)

            # Handle collisions
            self.simulation.handle_collisions()
            self.simulation.handle_boundary_collisions()

        positions = np.array(positions)  # Shape: [n_steps, num_particles, 3]
        velocities = np.array(velocities)  # Shape: [n_steps, num_particles, 3]
        return positions, velocities

    def simulate_gnn_predictions(self, model_wrapper: GNNModelWrapper, initial_positions, initial_velocities, n_steps=None):
        """
        Simulates dynamics using the trained GNN for n_steps starting from initial_positions and initial_velocities.

        Args:
            model_wrapper (GNNModelWrapper): Trained GNN model.
            initial_positions (np.ndarray): Initial positions of particles, shape [num_particles, 3].
            initial_velocities (np.ndarray): Initial velocities of particles, shape [num_particles, 3].
            n_steps (int, optional): Number of prediction steps. Defaults to self.config.NUM_ROLLOUT_STEPS.

        Returns:
            positions_pred (np.ndarray): Predicted positions over time, shape [n_steps, num_particles, 3].
        """
        if n_steps is None:
            n_steps = self.config.NUM_ROLLOUT_STEPS

        model_wrapper.model.eval()
        positions_pred = []
        current_positions = initial_positions.copy()
        current_velocities = initial_velocities.copy()

        for step in range(n_steps):
            positions_pred.append(current_positions.copy())

            # Prepare input tensor
            x = torch.tensor(np.hstack([current_positions, current_velocities]), dtype=torch.float32).to(self.device)  # Shape: [num_particles, 6]

            # Create edge index using radius_graph
            pos_tensor = torch.tensor(current_positions, dtype=torch.float32).to(self.device)
            edge_index = radius_graph(pos_tensor, r=self.config.INFLUENCE_RADIUS, loop=False)

            # Compute edge attributes (displacement vectors)
            row, col = edge_index
            edge_attr = pos_tensor[col] - pos_tensor[row]  # Shape: [num_edges, 3]

            # Create Data object
            data_input = Data(x=x, edge_index=edge_index, edge_attr=edge_attr).to(self.device)

            # Predict delta_x
            with torch.no_grad():
                delta_x = model_wrapper.model(data_input).cpu().numpy()  # Shape: [num_particles, 3]

            # Update positions and velocities
            current_positions += delta_x # * self.config.TIME_STEP
            current_velocities = delta_x / self.config.TIME_STEP #current_velocities = (current_positions - positions_pred[-1]) / self.config.TIME_STEP  # Estimate new velocities

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

            self.simulation.handle_collisions()
            self.simulation.handle_boundary_collisions()

            # Update positions and velocities after handling
            current_positions = np.array([p.position for p in temp_particles])
            current_velocities = np.array([p.velocity for p in temp_particles])

        positions_pred = np.array(positions_pred)  # Shape: [n_steps, num_particles, 3]
        return positions_pred

    def create_comparison_animation_fixed_axes(self, positions_gt, positions_pred, n_steps=None):
        if n_steps is None:
            n_steps = self.config.NUM_ROLLOUT_STEPS
        num_particles = positions_gt.shape[1]

        # Define colors for particles
        colors_gt = ['blue'] * num_particles
        colors_pred = ['orange'] * num_particles

        # Initialize figure
        fig = go.Figure()

        # Initialize traces for ground truth trajectories and current positions
        for i in range(num_particles):
            # Ground Truth Trajectory Line
            fig.add_trace(go.Scatter3d(
                x=[], y=[], z=[],
                mode='lines',
                line=dict(color=colors_gt[i], width=2),
                name=f'GT Trajectory {i+1}',
                showlegend=False
            ))
            # Ground Truth Current Position Marker
            fig.add_trace(go.Scatter3d(
                x=[], y=[], z=[],
                mode='markers',
                marker=dict(color=colors_gt[i], size=4),
                name=f'GT Current Position {i+1}',
                showlegend=False
            ))

            # GNN Prediction Trajectory Line
            fig.add_trace(go.Scatter3d(
                x=[], y=[], z=[],
                mode='lines',
                line=dict(color=colors_pred[i], width=2),
                name=f'Pred Trajectory {i+1}',
                showlegend=False
            ))
            # GNN Prediction Current Position Marker
            fig.add_trace(go.Scatter3d(
                x=[], y=[], z=[],
                mode='markers',
                marker=dict(color=colors_pred[i], size=4),
                name=f'Pred Current Position {i+1}',
                showlegend=False
            ))

        # Create frames
        frames_plotly = []
        for step in range(n_steps):
            frame_data = []
            for i in range(num_particles):
                # Ground Truth Trajectory Line
                frame_data.append(go.Scatter3d(
                    x=positions_gt[:step+1, i, 0],
                    y=positions_gt[:step+1, i, 1],
                    z=positions_gt[:step+1, i, 2],
                    mode='lines',
                    line=dict(color=colors_gt[i], width=2),
                    showlegend=False
                ))
                # Ground Truth Current Position Marker
                frame_data.append(go.Scatter3d(
                    x=[positions_gt[step, i, 0]],
                    y=[positions_gt[step, i, 1]],
                    z=[positions_gt[step, i, 2]],
                    mode='markers',
                    marker=dict(color=colors_gt[i], size=4),
                    showlegend=False
                ))

                # GNN Prediction Trajectory Line
                frame_data.append(go.Scatter3d(
                    x=positions_pred[:step+1, i, 0],
                    y=positions_pred[:step+1, i, 1],
                    z=positions_pred[:step+1, i, 2],
                    mode='lines',
                    line=dict(color=colors_pred[i], width=2),
                    showlegend=False
                ))
                # GNN Prediction Current Position Marker
                frame_data.append(go.Scatter3d(
                    x=[positions_pred[step, i, 0]],
                    y=[positions_pred[step, i, 1]],
                    z=[positions_pred[step, i, 2]],
                    mode='markers',
                    marker=dict(color=colors_pred[i], size=4),
                    showlegend=False
                ))

            frames_plotly.append(go.Frame(data=frame_data, name=f'Frame {step+1}'))

        # Assign frames
        fig.frames = frames_plotly

        # Set layout with fixed axes
        half_viewport = self.config.VIEWPORT_WORLD_SIZE / 2
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

        # Optionally, add annotations or other layout customizations here

        return fig


    def create_comparison_animation_html(self, positions_gt, positions_pred, n_steps=None):
        fig_comparison = self.create_comparison_animation_fixed_axes(positions_gt, positions_pred, n_steps)
        html_str = fig_comparison.to_html(full_html=False)
        return html_str

    def log_animation_to_wandb(self, html_str, epoch):
        try:
            wandb.log(
                {f"3D Trajectories": wandb.Html(html_str, inject=False)},
                step=epoch,
            )
        except Exception as e:
            print(f"Error logging to wandb: {e}")
            print("Skipping figure logging for this epoch.")

    def train(self):
        progress_bar = tqdm(range(self.config.NUM_EPOCHS), desc="Training Progress", unit="epoch")

        for epoch in progress_bar:
            epoch_start_time = time.time()
            total_loss = 0
            for batch in self.data_loader:
                loss = self.model_wrapper.train_step(batch)
                total_loss += loss
            self.model_wrapper.step_scheduler()
            current_lr = self.model_wrapper.get_learning_rate()

            # Update progress bar
            avg_loss = total_loss / len(self.data_loader)
            progress_bar.set_description(
                f"Epoch {epoch + 1}/{self.config.NUM_EPOCHS}, Loss: {avg_loss:.6f}, LR: {current_lr:.6f}"
            )

            # Log training metrics to wandb
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "loss": avg_loss,
                    "learning_rate": current_lr,
                },
                step=epoch + 1,
            )

            # Optionally, add training speed metrics
            epoch_duration = time.time() - epoch_start_time
            epochs_per_second = 1 / epoch_duration
            wandb.log({"epochs_per_second": epochs_per_second}, step=epoch + 1)

            # Visualization and Logging every WANDB_VISUALIZATION_FREQUENCY epochs
            if (epoch + 1) % self.config.WANDB_VISUALIZATION_FREQUENCY == 0 or epoch == 0:
                # Simulate ground truth
                positions_gt, velocities_gt = self.simulate_ground_truth(
                    self.test_positions_initial, self.test_velocities_initial, n_steps=self.config.NUM_ROLLOUT_STEPS
                )

                # Simulate GNN predictions
                positions_pred = self.simulate_gnn_predictions(
                    self.model_wrapper,
                    self.test_positions_initial,
                    self.test_velocities_initial,
                    n_steps=self.config.NUM_ROLLOUT_STEPS,
                )

                # Create animation HTML
                html_str = self.create_comparison_animation_html(positions_gt, positions_pred, n_steps=self.config.NUM_ROLLOUT_STEPS)

                # Log the HTML animation to wandb
                self.log_animation_to_wandb(html_str, epoch + 1)

        # Finish Weights & Biases Run
        wandb.finish()


# ========================
# Main Execution
# ========================
def main():
    # Initialize configuration
    config = Config()

    # Initialize and run the trainer
    trainer = Trainer(config)
    trainer.train()

    # Optionally, visualize the final simulation
    simulation = trainer.simulation
    visualizer = trainer.visualizer

    for step in tqdm(range(config.SIMULATION_STEPS), desc="Processing Timesteps"):
        positions, edges, widths, colliding_pairs = simulation.update()
        visualizer.generate_frame(positions, edges, widths, colliding_pairs, step)

    # Create and show the animation
    visualizer.create_animation()


if __name__ == "__main__":
    main()
