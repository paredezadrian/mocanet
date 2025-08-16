# MOCA-Net: Memory-Orchestrated Context Allocation Network

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

MOCA-Net represents a novel approach to neural network architecture design, combining sparse mixture-of-experts, external memory mechanisms, and budget-aware computation for efficient sequence modeling. The architecture introduces three key innovations: intelligent sparse token routing, learnable memory banks with adaptive gating, and differentiable budget optimization during training.

## Quick Start

Get up and running with MOCA-Net in just a few commands:

```bash
# Setup environment
make setup

# Run tests
make test

# Train on copy task (fast CPU run)
make train

# Run ablation studies
make ablate

# Generate plots
python scripts/plot_runs.py
```

## Architecture Overview

MOCA-Net's design philosophy centers around three core innovations that work together to achieve efficient sequence modeling:

1. **Sparse Token Router**: Dynamically selects which experts and memory slots to engage per token, operating under strict compute budget constraints
2. **External Memory Bank**: Implements learnable memory slots with sophisticated gated update mechanisms for long-term information retention
3. **Budget-Aware Training**: Incorporates a differentiable loss term that actively encourages efficient resource utilization throughout the training process

This architectural approach achieves O(L) complexity while preserving the expressive power needed for complex sequence modeling tasks through intelligent resource allocation strategies.

## System Architecture

The data flow through MOCA-Net follows this streamlined path:

```
Input → Token Router → [Experts + Memory] → Combined Output → Task Head
```

**Core Components:**
- **Token Router**: Intelligently routes tokens to top-k experts and memory slots based on learned routing policies
- **Expert Layer**: Implements a mixture of lightweight MLP experts with shared projection layers for efficiency
- **Memory Bank**: Provides external memory with attention-based read/write operations for persistent information storage
- **Budget Loss**: Continuously monitors and optimizes resource usage during training

For a comprehensive understanding of the mathematical foundations, refer to [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Performance Targets

MOCA-Net is designed to meet specific performance benchmarks across different tasks:

| Task | Metric | Target | CPU Runtime |
|------|--------|--------|-------------|
| Copy/Recall | Accuracy | ≥95% | ≤10 min |
| Text Classification | Accuracy | ≥80% | ≤5 min |

## Installation

### Prerequisites
- Python 3.12 or higher
- Ubuntu 24.04 or compatible Linux distribution

### Setup Instructions
```bash
# Clone the repository
git clone https://github.com/paredezadrian/mocanet.git
cd mocanet

# Create virtual environment and install dependencies
make setup

# Verify installation
make test
```

## Usage Examples

### Training Workflows

MOCA-Net provides several training configurations to suit different research needs:

```bash
# Train on copy task (default configuration)
make train

# Train on text classification task
make demo

# Quick training runs for rapid experimentation
make run-copy    # 1000 steps on copy task
make run-text    # 500 steps on text classification
```

### Model Evaluation

Evaluate your trained models using the built-in evaluation framework:

```bash
# Evaluate trained model on copy task
python -m mocanet.eval runs/mocanet_best.pt --task copy

# Generate comprehensive training plots
python scripts/plot_runs.py
```

### Ablation Studies

Explore the impact of different architectural components:

```bash
# Run comprehensive ablation studies
make ablate

# Results are automatically saved to runs/ablation/
```

## Configuration

MOCA-Net employs YAML-based configuration management powered by Hydra, enabling flexible experiment management and reproducible research. The configuration system is organized into logical components:

- **base.yaml**: Core model architecture and training parameters
- **copy_task.yaml**: Task-specific settings for copy/recall experiments
- **text_cls.yaml**: Configuration for text classification tasks

### Key Configuration Parameters

```yaml
model:
  embedding_dim: 128        # Token embedding dimension
  num_experts: 4            # Number of expert networks in mixture
  num_memory_slots: 64      # Memory bank capacity
  top_k_experts: 2          # Sparse routing parameter (top-k selection)
  router_temperature: 1.0   # Routing temperature for sparsity control

training:
  batch_size: 64            # Training batch size
  max_steps: 5000           # Maximum training steps
  learning_rate: 1e-3       # Learning rate
  warmup_steps: 200         # Learning rate warmup steps
  gradient_clip_norm: 1.0   # Gradient clipping norm
```

## Expected Results

### Copy Task Performance
- **Target Accuracy**: ≥95% on sequences up to 60 tokens
- **Expected Runtime**: ≤10 minutes on CPU
- **Memory Usage**: <4GB RAM

### Text Classification Performance
- **Target Accuracy**: ≥80% on synthetic SST-2 subset
- **Expected Runtime**: ≤5 minutes on CPU
- **Dataset Size**: 10,000 synthetic samples

## Ablation Studies

The framework supports systematic ablation studies to understand component contributions:

1. **No Memory**: Disables external memory bank to isolate memory effects
2. **No Experts**: Replaces mixture-of-experts with single expert architecture
3. **Dense Routing**: Uses all experts instead of sparse routing for comparison
4. **Smaller Model**: Reduces model size by half to analyze scaling effects

Execute ablation studies with: `make ablate`

## Testing

Ensure code quality and functionality with the comprehensive testing suite:

```bash
# Run complete test suite
make test

# Run specific test file with verbose output
python -m pytest tests/test_layers.py -v

# Generate coverage report
python -m pytest tests/ --cov=src/mocanet --cov-report=html
```

## Monitoring and Logging

Training progress is comprehensively logged using multiple tools:

- **Rich**: Provides beautiful, interactive console output with real-time progress bars
- **TensorBoard**: Tracks training curves, metrics, and model performance over time
- **Checkpoints**: Automatically saves model states every 1000 steps for recovery

## Troubleshooting

### Common Issues and Solutions

1. **CUDA Out of Memory**: Reduce batch size in configuration files
2. **Slow Training Performance**: Ensure `num_workers=0` for CPU-based training
3. **Import Errors**: Run `make setup` to properly install all dependencies

### Performance Optimization Tips

- Use `batch_size=16` for text classification tasks on CPU
- Set `max_steps=1000` for rapid experimentation cycles
- Enable `gradient_clip_norm=1.0` for training stability

## Future Development Roadmap

The MOCA-Net project continues to evolve with planned enhancements:

1. **Learned Write Policy**: Develop adaptive memory update strategies based on input characteristics
2. **KV Compression**: Implement efficient memory representation techniques
3. **Hierarchical Experts**: Design multi-level expert organization for complex tasks
4. **Retrieval-Augmented Tasks**: Integrate external knowledge sources
5. **Dynamic Routing**: Create adaptive expert selection mechanisms based on input complexity

## Research References

MOCA-Net builds upon and extends several foundational works in neural architecture design:

- **Mixture of Experts**: [Shazeer et al. (2017)](https://arxiv.org/abs/1701.06538) - Out of the Box: An Empirical Study of the Real-World Effectiveness of Neural Machine Translation
- **External Memory**: [Graves et al. (2014)](https://arxiv.org/abs/1410.5401) - Neural Turing Machines
- **Sparse Routing**: [Lepikhin et al. (2020)](https://arxiv.org/abs/2006.16668) - GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding

## Contributing

We welcome contributions from the research community! To contribute:

1. Fork the repository
2. Create a feature branch for your contribution
3. Add comprehensive tests for new functionality
4. Ensure all tests pass: `make test`
5. Submit a pull request with detailed description

## License

This project is licensed under the **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License** (CC BY-NC-SA 4.0).

**License Terms:**
- ✅ **You can share and adapt** this work freely
- ✅ **You must give attribution** to the original author
- ❌ **You cannot use it for commercial purposes**
- ✅ **You must share adaptations** under the same license

For complete license details, see the [LICENSE](LICENSE) file, or visit [Creative Commons](https://creativecommons.org/licenses/by-nc-sa/4.0/) for more information.

## Acknowledgments

MOCA-Net stands on the shoulders of the open-source machine learning community:

- **PyTorch Team**: For providing an excellent deep learning framework
- **Rich Library**: For beautiful console output and user experience
- **Hydra**: For robust configuration management and experiment tracking
- **Open-Source ML Community**: For continuous inspiration and collaboration

---

**Ready to explore the frontiers of efficient neural architecture design? Begin your journey with `make setup` and discover the innovative world of MOCA-Net.**
