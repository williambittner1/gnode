# Objective

---

Train a model to learn the physics of object dynamics (falling and collisions) from multi-view video datasets. Use a Hierarchical Message Passing Graph Neural ODE (Hierarchical GNODE) to predict future 3D Gaussian parameters. At test time, the model should:

- Interpolate or extrapolate the 3D scene in time.
- Generate novel views of the scene.

# **Model Overview**

## **1. Dataset:**

---

- `num_sequences` sequences of a physically simulated scene with `num_timesteps` timesteps rendered/captured from multi-view.
- Scenes:
    - orbital dynamics (Blender, in-python simulation)
    - double-pendulum (Blender)
    - rigid-object interactions (Blender, Kubric Dataset Generator, in-python simulation)
    - dice thrown (Blender, in-python simulation)
- prepare dataset by 3DGS encoding each timestep separately to obtain the following per-timestep input data:
    - multi-view rendering
    - optical-flow from rendering
    - pixel-wise semantic features from 2D foundational model
    - Pseudo-GT 3D Gaussians (without temporal correspondences) obtained independently for each timestep from many (e.g. 100+) views with distilled features (from optical flow and 2D foundational model)

## **2. Core Model:**

---

### **Encoding Multi-View 2D Video Data to 3D Gaussians**

2D multi-view image data is encoded (independently for every timestep for now) into 3D Gaussians with latent feature embeddings (distilled via a 2D Foundational Model) represented as a graph with node and edge attributes through the following steps:

1. **Feature Extraction via 2D Foundation Models (in 2D image-space):**
    - Use 2D foundational models such as CLIP, SAM, or DINO to extract frame-level semantic features.
    - Extract object-level features (e.g., positions, sizes, and semantic labels) using object segmentation and detection models like SAM.
    - Generate dense embeddings for each view, capturing both appearance and motion information.
2. **Feature Distillation Across Views (2D images → 3D Gaussians)**
    
    Apply feature 3D Gaussian Splatting (Feature-3DGS) to distill 2D multi-view image features (semantic CLIP, SAM, DINO features or optical flow features) to 3D Gaussians with latent features. 
    

### **Hierarchical Graph Neural ODE**

> The Hierarchical GNODE processes the graph using hierarchical message passing to model both local and global interactions. Predict Gaussians at the next timestep by solving and integrating (`torchdiffeq` ode-solver) the dynamics with a Hierarchical Graph Neural ODE framework.
> 
1. **Hierarchical Graph Construction (3D Gaussians → Hierarchical Graph)**
    
    <aside>
    💡
    
    The **simple (non-hierarchical) graph** has:
    
    - **node attributes** represent a Gaussian particle**:**
        - 3D-Position (derived from 3D Gaussian Splatting optimization)
        - 3D-Velocity (derived from optical flow)
        - Semantic/Latent features like mass, semantics,… **(**derived from learned embeddings e.g. via feature distillation as in Feature-3DGS from 2D Foundation Models like SAM)
    - **edge attributes** model interactions between particles**:**
        - pairwise distances,
        - relative positions/velocities and
        - relational features (such as rigidity or interaction strength derived from embeddings).
    </aside>
    
    Hierarchical Graphs are constructed to organize nodes at different **levels of abstraction**:
    
    - **Level 1 (Micro):** Nodes represent individual particles (e.g., Gaussian parameters like position, velocity, and latent features).
    - **Level 2 (Meso):** Abstract nodes group particles based on spatial or feature similarity (e.g., clustering particles within spatial regions or shared semantic features).
    - **Level 3 (Macro):** High-level nodes aggregate meso-level representations, capturing global scene properties or dynamics.
    
    Connections between levels are represented as inter-level edges, while intra-level edges model interactions within the same level.
    
    <aside>
    <img src="/icons/command-line_gray.svg" alt="/icons/command-line_gray.svg" width="40px" />
    
    **Construction Steps:**
    
    1. Clustering for Super Nodes:
        - At each level, group lower-level nodes into super nodes using clustering algorithms (e.g., k-means in a hyper-space (scaled dimensions in 3d-space, 3d-velocity, nd-semantic/latent feature,…) or community detection like Louvain).
        - Each super node aggregates features (e.g., mean or weighted pooling) from its member nodes.
    2. Edge Creation:
        - **Intra-Level Edges:** Connect nodes within the same level based on spatial proximity (in hyper-space) or similarity metrics.
        - **Inter-Level Edges:** Connect nodes in lower levels to their corresponding super nodes in higher levels.
    </aside>
    
2. **Hierarchical Message Passing Schemes** 
    
    Message passing alternates between three schemes: **Bottom-Up**, **Within-Level**, and **Top-Down**.
    
    - **Bottom-Up Propagation (Super Node Update)**
        - Aggregates fine-grained information from lower-level nodes to higher-level super nodes.
        - Captures detailed local interactions and transmits them to broader contexts.
        - Update rule for a super node $s_i^t$ in level $t$:
            
            $$
            \mathbf{h}_{s^t_i} = \text{Aggregate}\left(\{\mathbf{h}_{v^{t-1}_j} : v^{t-1}_j \in \text{group}(s^t_i)\}\right)
            
            $$
            
            where **`Aggregate`** can be mean, sum, or attention-based pooling.
            
    - **Within-Level Propagation (Node Update):**
        - Applies flat GNN message passing within a single level.
        - Update rule for a node $v_i$ at layer $l$:
        
        $$
           \mathbf{h}_i^{(\ell+1)} = \sigma\left(W \cdot \text{Aggregate}\left(\{\mathbf{h}_j^{(\ell)} : j \in \mathcal{N}(i)\}\right)\right)
        
        $$
        
        - Performs flat GNN message passing within each level, allowing local interactions to refine node representations.
    - **Top-Down Propagation (Node Update with Global Context):**
        - Transmits global and meso-level information from higher-level super nodes to lower-level nodes.
        - Enhances fine-grained node representations with global context.
        - Update rule for a node $v_i^{t-1}$:
        
        $$
           \mathbf{h}_{v^{t-1}_i} = \text{Combine}\left(\mathbf{h}_{v^{t-1}_i}, \mathbf{h}_{s^t_j} : s^t_j \in \text{parent}(v^{t-1}_i)\right)
        
        $$
        

1. **Temporal Dynamics (Node Embedding Evolution):**
    
    Temporal evolution of node representations is modeled with Graph Neural ODEs. The GNODE integrates hierarchical features over time to predict future states of Gaussian particles.
    
    $$
       \frac{d\mathbf{h}_i}{dt} = f_{\theta}(\mathbf{h}_i, \mathbf{h}_j, \mathbf{e}_{ij})
    
    $$
    
    <aside>
    <img src="/icons/command-line_gray.svg" alt="/icons/command-line_gray.svg" width="40px" />
    
    **Integration:**
    
    1. **Initial Condition:** Use the output of hierarchical message passing as the initial state **h**(*t*=0).
        
        h(t=0)
        
    2. **ODE Solver:** Integrate dynamics forward in time using solvers like RK4 or Dormand-Prince.
    3. **Recursive Rollout:** For multi-step prediction, recursively apply the dynamics model.
    </aside>
    

### **3D Gaussian Splatting Differentiable Rasterizer**

- Render photorealistic novel-views of Gaussians.
- Compute differentiable image-based indirect loss.

<aside>
<img src="/icons/command-line_gray.svg" alt="/icons/command-line_gray.svg" width="40px" />

### **Summary**

1. **Encoding Multi-View 2D Video Data to 3D Gaussians with Features**
    - Feature Extraction via 2D Foundation Models 
    (in 2D image-space)
    - Encode multi-view video data into 3D Gaussian particles with latent features through feature distillation across views 
    (2D images → 3D Gaussians)
2. **Graph Construction:**
    - Construct a hierarchical graph with nodes, edges, and inter-level connections.
3. **Hierarchical Message Passing:**
    - Apply bottom-up propagation to transmit fine-grained details to higher levels.
    - Perform within-level propagation to refine local interactions.
    - Use top-down propagation to integrate global context back to detailed representations.
4. **Temporal Dynamics:**
    - Evolve node states using Graph Neural ODEs for dynamic predictions.
    - Combine hierarchical representations and temporal evolution for long-horizon prediction.
5. **Output:**
    - Predicted Gaussian parameters for the next timestep.
    - Rendered frames using a differentiable 3D Gaussian Splatting Rasterizer.
</aside>

## **3. Training Framework**

---

- **Input:**
    - Multi-view video frames at a single or multiple timesteps.
    - Optical flow between timesteps for all views.
    - Ground truth (GT) Gaussians representing object states acquired through 3DGS Encoding.
- **Output:**
    - Predicted Gaussians at the next timestep (GNODE)
    - Rendered images from the predicted Gaussians.
- **Loss Computation:**
    - **Indirect Loss** on Images**:** 
    Compare rendered frames to ground truth multi-view frames (pixel-wise loss, perceptual loss).
    - **Direct Loss** on predicted Dynamic Graph**:** 
    Mean Squared Error (MSE) between predicted and ground truth Gaussian parameters. (e.g., node-attributes: position, size, intensity, color,… edge-attributes: particle-distance/rigidity,…). Physics consistency loss by adding penalties for physically inconsistent predictions, such as overlapping Gaussians or unrealistic velocities.
- **Gradual Prediction Rollout:**
    - In the **initial phase (short rollout)** start by training the GNODE to predict **one timestep ahead** using the input data. Use the predicted Gaussians at *t*+1 to compute the loss against the ground truth.
    - In the **progressive rollout phase (longer horizons)** gradually increase the number of predicted timesteps (rollout length) during training from one predicted Gaussians at t+1 to multiple timesteps [t+1, t+K]. Combine losses at all predicted timesteps. Potentially use irregularly sampled timesteps for loss computation.
    - Optionally: Test encoding multiple timesteps using an Encoder to produce the rollout instead of conditioning only on a single input timestep. Would give the option to refine node/edge attributes that should remain constant over time like e.g. edge rigidity constraint between nodes. An attribute that can be conditioned on a dynamic graph.
- **Gradual Input Expansion *(needs more brainstorming)***
    - Encode multiple timesteps using an encoder to produce the rollout instead of conditioning only on a single input timestep.
    - Requires to extend Feature 3DGS Encoder that provides Feature 3D Gaussians that are then converted to a static Hierarchical GNODE to a Feature 4DGS Encoder providing Feature 4D Gaussians that can be converted to a Hierarchical GNODE. Would give the option to refine node/edge attributes that should remain constant over time (like e.g. edge rigidity constraint between nodes. An attribute that can be conditioned on a dynamic graph).
    - The GNODE learns to leverage richer temporal context as training progresses.
    - Reduces reliance on single-timestep dynamics, enabling better extrapolation.
- **Noise Injection During Training**
    - Begin noise injection after the rollout length exceeds a threshold (e.g. 3 timesteps)
    - Add Gaussian noise to the predicted Gaussians before feeding them autoregressively as inputs for the next prediction.
    - Noise injection during both input aggregation/encoding and rollout prediction might reduce error accumulation.

# Project Code Directory Organization

---

```markdown
project-root/
├── data/
│   ├── dataset_1/
│   │   ├── sequence_1/
│   │   │   ├── rgb/
│   │   │   │   ├── rgb_cam1_timestep1.png
│   │   │   │   ├── rgb_cam1_timestep2.png
│   │   │   │   └── ...
│   │   │   ├── opticalflow_gt/
│   │   │   ├── opticalflow_estimated/
│   │   │   ├── semantic_segmentation_gt/
│   │   │   ├── sam_features/
│   │   │   ├── 3d_feature_gaussians/ (num_timesteps gaussian models)
│   │   │   ├── static_graphs/ (num_timesteps graphs)
│   │   │   └── ...
│   │   ├── sequence_2/
│   │   └── ...
│   ├── dataset_2/
│   └── ...
├── models/
│   ├── feature_extractor.py
│   ├── gaussian_encoder.py
│   ├── graph_constructor.py
│   └── ...
├── scripts/
│   ├── extract_features.py
│   ├── encode_gaussians.py
│   └── construct_graphs.py
├── configs/
│   └── config.yaml
└── README.md

```

# **Project Timeline and Roadmap**

---

### **Phase 1: Core Model Development (2nd December 2024 – 31st December 2025)**

**Objective:** Implement the first prototypical core model to play with some toy data (3D Pointclouds, small rgb-image dataset)

- **Feature-3DGS Encoding**
Build Feature 3D Gaussian Encoding from Multi-View Images (+ CLIP/SAM/DINO Encodings and Optical Flow)
- **Graph Construction from Feature 3D Gaussians**
Build hierarchical graphs with: Micro (particle-level), meso (cluster-level), and macro (global) nodes and Intra- and inter-level edges.
- **Hierarchical Message Passing Graph Neural ODE**
Implement hierarchical Message Passing GNODE
- **Visualization Tools**
Implement Visualization / Logging Tools
    
    

### **Phase 2: Dataset Preparation (1st January - 15 January)**

**Objective:** set-up physics simulation, render multi-view sequences (rgb, optical flow, gt camera positions), precompute image features.

- Set up synthetic **physics-simulation**:
    - Orbital dynamics (Blender or pure-python simulation)
    - Double pendulum (Blender)
    - Rigid-object interactions (Kubric Dataset Generator, Blender)
    - Dice throws (Blender)
    - Balls rolling on a surface (Blender)
- **Render multi-view image frames** (rgb, pixel-wise optical flow, pixel-wise image segmentation) for each scene
    
    Per dataset (orbital dynamics, double pendulum, rigid-object interactions, dice throws, balls rolling) render:
    
    - 50+ views (rgb, gt optical flow, gt image segmentation) per timestep for pseudo-GT Gaussian creation
    - 1000+ timesteps
    - 1000+ sequences
- **Compute 2D Features (CLIP, SAM, DINO) and Optical Flow (RAFT)**
    
    Per image frame extract 2D semantic features using CLIP, SAM, or DINO
    
    Per image frame compute optical flow using RAFT
    
- **Generate pseudo-GT Feature 3D Gaussians**
    
    Per timestep generate pseudo-GT Feature 3D Gaussians via Feature 3DGS optimization distilling image features either from:
    
    - gt optical flow and gt image segmentation or
    - computed optical flow and computed 2D semantic CLIP/SAM/DINO features)
    - No temporal correspondences of gaussians between timesteps!

### **Phase 3: Training and Experimentation (15th January - 31st January)**

**Objective:** Train Hierarchical GNODE and perform baseline comparisons.

**Initial Training** 

- Train Hierarchical GNODE with:
    - Single-timestep predictions
    - Gradually increase prediction horizons (t+1 to t+K)
    - Irregularly sampled timesteps to predict
    - Gradual Input Expansion (Extend Feature 3DGS Encoder to a Feature 4DGS Encoder)
    - Introduce noise injection for robustness
    - Test stability under perturbed initial conditions
- Train on simplest dataset first
- Evaluation/Inference:
    - Condition on single timestep / multiple timesteps
    - Direct Loss: MSE for Gaussian parameters (only possible for dataset with corresponding GT Gaussians between timesteps, pseudo-gt gaussians with no temporal correspondences aren’t sufficient).
    - Indirect Loss: Image-based Loss (MSE, LPIPS, SSIM,…)
    - Temporal consistency over Extrapolation Length (Accumulative Loss over Rollout-Length)

### Phase 4: **Full Pipeline Integration and Validation (1st February - 15th February)**

**Objective:** Finalize the end-to-end pipeline and validate on all synthetic datasets with extensive tests.

- If time requires expand with real-world dataset (would require implementing dataset preparation pipeline for real-world data)
- Test interpolation, extrapolation, and novel view synthesis
- Perform ablational studies on
    - different 3D Backbones: PointNet, MLP, Attention-Based Transformer operating directly on 3D Gaussian Dynamics
    - contribution of hierarchical message passing (hierarchical component)
    - role of temporal GNODE dynamics (Neural ODE component)

### Phase 5: **Baseline Setup (15th February - 28th February)**

**Objective:** Implement baseline models and perform baseline experiments.

- 2D-based methods: Video Extrapolation (Phyworld, VidODE, ExtDM)
- 3D-based methods: Temporal Scene Extrapolation (Gaussian Prediction (no code available yet), Learning 3D Particle-based Simulators from RGB-D Videos (no code available yet))
- Different 3D Gaussian Backbone (ours: Hierarchical GNODE; others: PointNet, MLP)

### Phase 6: **Paper Writing and Final Submission (1st March - 8th March)**

**Objective:** Prepare and submit the ICCV 2025 paper.

1. **Draft Writing (1st – 4th March 2025)**
    
    Introduction, Related Work, Methodology, Experiments, and Results including:
    
    - Qualitative results for dynamics and rendering
    - Quantitative metrics for all baseline comparisons
2. **Visualization and Final Edits (5th – 7th March 2025)**
    
    Generate visualizations:
    
    - Gaussian trajectories, rollout predictions, novel views
    - Final proofing and formatting
3. **Project-Demo Page and Github Code Repo**
    - Set up a project demo page with explanatory material and visualizations
    - Push clean Github code repo
4. **Submission (8th March 2025)**
    
    Submit the finalized paper and supplementary materials to ICCV 2025.