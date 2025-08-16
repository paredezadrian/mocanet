# MOCA-Net Architecture

## Overview

MOCA-Net (Memory-Orchestrated Context Allocation Network) is a novel neural architecture that combines three key ideas:

1. **Sparse Mixture of Experts (MoE)** with dynamic routing
2. **External Memory Bank** with learnable keys and values
3. **Budget-Aware Computation** with differentiable resource constraints

## Architecture Diagram

```
Input Sequence: [x₁, x₂, ..., xₜ]
       ↓
Token Embeddings + Position Embeddings
       ↓
┌─────────────────────────────────────────────────────────────┐
│                    TOKEN ROUTER                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    │
│  │Expert Router│    │Memory Router│    │Budget Pred. │    │
│  │   (MLP)     │    │   (MLP)     │    │   (MLP)     │    │
│  └─────────────┘    └─────────────┘    └─────────────┘    │
│       ↓                   ↓                   ↓            │
│  [w₁, w₂, ..., wₖ]  [m₁, m₂, ..., mₘ]   budget_usage    │
└─────────────────────────────────────────────────────────────┘
       ↓                   ↓
┌─────────────┐    ┌─────────────┐
│  EXPERT     │    │   MEMORY    │
│   LAYER     │    │    BANK     │
│             │    │             │
│  ┌─────┐    │    │  ┌─────┐    │
│  │ E₁  │    │    │  │ M₁  │    │
│  └─────┘    │    │  └─────┘    │
│  ┌─────┐    │    │  ┌─────┐    │
│  │ E₂  │    │    │  │ M₂  │    │
│  └─────┘    │    │  └─────┘    │
│     ...     │    │     ...     │
│  ┌─────┐    │    │  ┌─────┐    │
│  │ Eₖ  │    │    │  │ Mₘ  │    │
│  └─────┘    │    │  └─────┘    │
└─────────────┘    └─────────────┘
       ↓                   ↓
   Expert Output        Memory Output
       ↓                   ↓
       └───────┬───────────┘
               ↓
        Combined Output
               ↓
        Output Projection
               ↓
        Task-Specific Head
               ↓
           Output
```

## Mathematical Formulation

### 1. Token Routing

For each token embedding $h_t \in \mathbb{R}^d$ at position $t$:

**Expert Routing:**
$$r_t^e = \text{softmax}\left(\frac{W_e h_t + b_e}{T}\right)$$
$$w_t^e = \text{top-k}(r_t^e, k)$$

**Memory Routing:**
$$r_t^m = \text{softmax}\left(\frac{W_m h_t + b_m}{T}\right)$$
$$w_t^m = \text{top-k}(r_t^m, k_m)$$

**Budget Prediction:**
$$b_t = \sigma(W_b h_t + b_b)$$

Where:
- $T$ is the temperature parameter for sparsity
- $k$ is the number of experts to route to
- $k_m$ is the number of memory slots to access
- $\sigma$ is the sigmoid activation

### 2. Memory Bank Operations

**Memory Read:**
$$Q_t = h_t \in \mathbb{R}^d$$
$$K = [k_1, k_2, ..., k_M] \in \mathbb{R}^{M \times d}$$
$$V = [v_1, v_2, ..., v_M] \in \mathbb{R}^{M \times d}$$

Attention scores:
$$s_{t,i} = \frac{Q_t^T K_i}{\sqrt{d}} \cdot w_{t,i}^m$$

Memory output:
$$m_t = \sum_{i=1}^M \text{softmax}(s_t)_i \cdot V_i$$

**Memory Update (Optional):**
$$g_t = \sigma(W_g [m_t; h_t] + b_g)$$
$$v_i^{new} = g_t \cdot h_t + (1 - g_t) \cdot v_i \cdot \gamma$$

Where $\gamma$ is the memory decay factor.

### 3. Expert Layer

**Input Projection:**
$$h_t' = W_{in} h_t + b_{in}$$

**Expert Computation:**
$$e_{t,i} = f_i(h_t') \quad \text{for } i \in \{1, 2, ..., k\}$$

**Output Combination:**
$$o_t = W_{out} \left(\sum_{i=1}^k w_{t,i}^e \cdot e_{t,i}\right) + b_{out}$$

### 4. Budget Loss

**Resource Usage:**
$$u_t^e = \frac{1}{K} \sum_{i=1}^K \mathbb{I}[w_{t,i}^e > 0]$$
$$u_t^m = \frac{1}{M} \sum_{i=1}^M \mathbb{I}[w_{t,i}^m > 0]$$

**Budget Loss:**
$$\mathcal{L}_{budget} = \lambda \left(\text{ReLU}(u_t - \tau) + \text{MSE}(b_t, u_t)\right)$$

Where:
- $u_t = (u_t^e + u_t^m) / 2$ is the total resource usage
- $\tau$ is the target budget threshold
- $\lambda$ is the budget loss weight

## Algorithmic Details

### Forward Pass Algorithm

```
Algorithm: MOCA-Net Forward Pass
Input: Input sequence X = [x₁, x₂, ..., xₜ]
Output: Output sequence Y = [y₁, y₂, ..., yₜ]

1. Token Embedding
   for t = 1 to T:
       h_t = Embed(x_t) + PosEmbed(t)

2. Token Routing
   for t = 1 to T:
       (w_t^e, w_t^m, _, _, b_t) = Router(h_t)

3. Memory Operations
   for t = 1 to T:
       m_t = MemoryBank(h_t, w_t^m, h_t)  # Read + Update

4. Expert Processing
   for t = 1 to T:
       e_t = ExpertLayer(h_t, w_t^e)

5. Output Generation
   for t = 1 to T:
       o_t = LayerNorm(Dropout(W_out(m_t + e_t)))
       y_t = TaskHead(o_t)

6. Budget Loss Computation
   L_budget = BudgetLoss(w_t^e, w_t^m, b_t)

return Y, L_budget
```

### Training Algorithm

```
Algorithm: MOCA-Net Training
Input: Training data D, model parameters θ
Output: Trained model

1. Initialize model parameters θ
2. for epoch = 1 to max_epochs:
       for batch in D:
           # Forward pass
           Y_pred, L_budget = MOCANet(X_batch)
           
           # Task loss
           L_task = TaskLoss(Y_pred, Y_true)
           
           # Total loss
           L_total = L_task + L_budget
           
           # Backward pass
           ∇θ = ∇_θ L_total
           
           # Gradient clipping
           ∇θ = clip(∇θ, norm_threshold)
           
           # Parameter update
           θ = θ - lr * ∇θ
           
           # Memory bank update (if enabled)
           UpdateMemoryBank()
```

## Key Design Principles

### 1. Sparsity First
- Only activate necessary experts and memory slots per token
- Top-k routing ensures computational efficiency
- Temperature scaling controls sparsity level

### 2. Memory Persistence
- External memory bank maintains long-term information
- Gated updates prevent catastrophic forgetting
- Decay factor ensures stable training

### 3. Budget Awareness
- Differentiable budget loss encourages efficient resource usage
- Predictable computation patterns for deployment
- Configurable resource constraints

### 4. Modularity
- Easy to disable components for ablation studies
- Configurable expert and memory counts
- Flexible routing strategies

## Computational Complexity

- **Time Complexity**: O(T × (d² + K × d + M × d))
  - T: sequence length
  - d: embedding dimension
  - K: number of experts
  - M: number of memory slots

- **Space Complexity**: O(T × d + K × d + M × d)
  - Linear in sequence length
  - Constant in expert and memory counts

## Advantages

1. **Efficiency**: Sparse routing keeps compute linear
2. **Memory**: Explicit long-term storage without attention overhead
3. **Flexibility**: Configurable resource usage and model size
4. **Interpretability**: Clear routing decisions and memory access patterns
5. **Scalability**: Easy to scale experts and memory independently

## Limitations

1. **Complexity**: More complex than standard architectures
2. **Tuning**: Requires careful tuning of routing parameters
3. **Memory**: External memory bank increases memory footprint
4. **Training**: Budget loss can be sensitive to hyperparameters
