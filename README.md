# Comprehensive Project Layout with Latent Node and Edge Features

This layout describes the complete pipeline for developing a **Graph Neural ODE (GNODE)** model for dynamic scene modeling, integrating latent node and edge features distilled from 2D foundational models like CLIP or DINO.

---

## 1. Project Overview

### Objective
- Learn the physics of dynamic 3D scenes from a multi-view dataset.
- Use a **Graph Neural ODE (GNODE)** to predict future Gaussian parameters representing object states.
- Incorporate latent features derived from 2D foundational models to enrich node and edge representations, improving model robustness and generalization.
- At test time, interpolate and extrapolate object dynamics and generate novel views of the scene.

---

## 2. Project Components

### 2.1 Input Data
1. **Multi-View Video Data:**
   - Calibrated multi-camera setup providing synchronized frames for each timestep.
   - Includes simple object scenes involving falling, colliding, and interacting dynamics.

2. **Precomputed Features:**
   - **Optical Flow:** Capture motion between consecutive frames.
   - **Ground Truth Gaussians:** Represent object states in each timestep with parameters such as:
     - Position \((x, y, z)\)
     - Velocity \((v_x, v_y, v_z)\)
     - Covariance matrix or size/intensity descriptors.
   - **Latent Features from Foundational Models:**
     - Extract image embeddings for each frame using a 2D foundational model (e.g., CLIP, DINO).
     - Distill object-level features (e.g., object embeddings, appearance features) for nodes.
     - Distill pairwise features (e.g., relational or contextual features) for edges.

---

### 2.2 Graph Representation
1. **Nodes:**
   - Represent Gaussian particles.
   - Node features include:
     - Gaussian parameters (position, velocity, size).
     - Latent embeddings distilled from 2D models (object-level features).
2. **Edges:**
   - Capture relationships between particles.
   - Edge features include:
     - Relative positions, velocities.
     - Latent features distilled from 2D models (relational/contextual features).

---

### 2.3 Graph Neural ODE (GNODE)
1. **Core Dynamics:**
   - A message-passing mechanism updates node embeddings using information from neighbors.
   - The GNODE models continuous-time dynamics as:
     \[
     \frac{d\mathbf{h}_i}{dt} = f_\theta(\mathbf{h}_i, \mathbf{h}_j, \mathbf{e}_{ij}, t)
     \]
     where:
     - \(\mathbf{h}_i\): Node embedding for Gaussian \(i\), including latent features.
     - \(f_\theta\): Neural network parameterizing the dynamics.
2. **ODE Solver:**
   - A differentiable ODE solver integrates the dynamics to predict future node states.

---

### 2.4 Rendering Pipeline
- Use a **differentiable rendering module** to generate multi-view frames from predicted Gaussians.
- Rendered frames are used for loss computation and visualization.

---

## 3. Training Framework

### 3.1 Training Schedule
| **Phase**            | **Inputs**              | **Rollout Horizon (\(K\))** | **Noise Injection** |
|-----------------------|-------------------------|-----------------------------|----------------------|
| Initial Phase         | Single timestep         | 1                           | None                 |
| Intermediate Phase    | Single timestep         | Gradually increase (\(K=2,3,\ldots\)) | Increasing (\(\sigma^2 = 0.01 \cdot K\)) |
| Advanced Phase        | Multi-timestep          | Maximum horizon (\(K_\text{max}\)) | High (\(\sigma^2 = 0.1\)) |

---

### 3.2 Training Workflow

#### **Input Preparation**
1. Extract object-level and relational latent features from 2D foundational models for each timestep.
2. For each batch:
   - Extract video frames, optical flow, and GT Gaussians.
   - Combine these with latent features to construct the graph.

#### **Graph Construction**
1. **Nodes:**
   - Initialize node features with:
     - Gaussian parameters (position, velocity, size).
     - Latent embeddings (object-level features).
2. **Edges:**
   - Compute edge features using:
     - Relative spatial information.
     - Latent embeddings (relational features).

#### **Rollout Prediction**
1. Use GNODE to iteratively predict Gaussians for \(t+1, t+2, \ldots, t+K\).
2. Inject noise into predictions for \(t+k > 1\):
   \[
   \mathbf{g}_{t+k}^\text{noisy} = \mathbf{g}_{t+k}^\text{pred} + \mathcal{N}(0, \sigma^2)
   \]

#### **Rendering**
- Render predicted Gaussians into frames for all \(K\) timesteps.

#### **Loss Computation**
1. **Gaussian Prediction Loss:**
   \[
   \mathcal{L}_\text{rollout} = \frac{1}{K} \sum_{k=1}^K \| \mathbf{g}_{t+k}^\text{pred} - \mathbf{g}_{t+k}^\text{gt} \|^2
   \]
2. **Rendering Loss:**
   \[
   \mathcal{L}_\text{render} = \frac{1}{K} \sum_{k=1}^K \| \mathbf{I}^\text{pred}_{t+k} - \mathbf{I}^\text{gt}_{t+k} \|^2
   \]
3. **Temporal Consistency Loss:**
   \[
   \mathcal{L}_\text{temporal} = \frac{1}{K-1} \sum_{k=1}^{K-1} \| \mathbf{g}_{t+k+1}^\text{pred} - \mathbf{g}_{t+k}^\text{pred} \|^2
   \]
4. **Regularization Loss:**
   - Penalize physically implausible behavior (e.g., overlapping Gaussians).

#### **Backward Pass and Optimization**
- Combine all losses:
  \[
  \mathcal{L} = \lambda_1 \mathcal{L}_\text{rollout} + \lambda_2 \mathcal{L}_\text{render} + \lambda_3 \mathcal{L}_\text{temporal} + \lambda_4 \mathcal{L}_\text{reg}
  \]
- Update model parameters using a gradient-based optimizer (e.g., Adam).

---

## 4. Testing Framework

### Test-Time Objectives
1. **Interpolation:** Predict intermediate frames between given timesteps.
2. **Extrapolation:** Extend predictions forward or backward in time.
3. **Novel View Synthesis:** Generate frames from unseen viewpoints using predicted Gaussians.

### Metrics
1. **Gaussian Parameter Accuracy:** Errors in positions, velocities, and other parameters.
2. **Frame Reconstruction Accuracy:** PSNR, SSIM.
3. **Temporal Coherence:** Smoothness of predictions.

---

## 5. Implementation Modules

### 5.1 Data Processing
- Extract optical flow, ground truth Gaussians, and latent features from 2D models.

### 5.2 Graph Construction
- Build dynamic graphs with latent node and edge features.

### 5.3 GNODE Architecture
- Integrate GNN-based message passing with Neural ODE dynamics and latent features.

### 5.4 Rendering Module
- Differentiable rendering for multi-view frames.

### 5.5 Loss Computation
- Implement all loss terms.

---

## 6. Roadmap

### Phase 1: Core Development (Weeks 1–4)
- Implement GNODE architecture and graph construction.
- Test single-step prediction with latent features.

### Phase 2: Gradual Rollout (Weeks 5–8)
- Add progressive rollout prediction and noise injection.

### Phase 3: Full Pipeline Integration (Weeks 9–12)
- Integrate rendering, latent features, and loss computation.

### Phase 4: Testing and Refinement (Weeks 13–16)
- Test interpolation, extrapolation, and novel view synthesis.

---

## 7. Expected Outcomes
1. A trained GNODE capable of leveraging latent features for accurate and robust dynamics modeling.
2. High-quality interpolation, extrapolation, and novel view synthesis results.
3. Comprehensive evaluation with quantitative metrics and qualitative visualizations.
