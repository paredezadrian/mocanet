# MOCA-Net: Memory-Orchestrated Context Allocation Network

A novel neural network architecture that combines sparse mixture-of-experts, external memory, and budget-aware computation for efficient sequence modeling.

## 🚀 Quick Start

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

## 📋 What Makes MOCA-Net Novel?

MOCA-Net introduces three key innovations:

1. **Sparse Token Router**: Dynamically selects which experts and memory slots to use per token under compute budget constraints
2. **External Memory Bank**: Learnable memory slots with gated updates for long-term information storage
3. **Budget-Aware Training**: Differentiable loss term that encourages efficient resource usage

This architecture achieves **O(L) complexity** while maintaining expressiveness through intelligent resource allocation.

## 🏗️ Architecture Overview

```
Input → Token Router → [Experts + Memory] → Combined Output → Task Head
```

- **Token Router**: Routes tokens to top-k experts and memory slots
- **Expert Layer**: Mixture of lightweight MLP experts with shared projections
- **Memory Bank**: External memory with attention-based read/write operations
- **Budget Loss**: Encourages efficient resource usage during training

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed mathematical formulation.

## 📊 Performance Targets

| Task | Metric | Target | CPU Runtime |
|------|--------|--------|-------------|
| Copy/Recall | Accuracy | ≥95% | ≤10 min |
| Text Classification | Accuracy | ≥80% | ≤5 min |

## 🛠️ Installation

### Prerequisites
- Python 3.12+
- Ubuntu 24.04 (or compatible Linux)

### Setup
```bash
# Clone repository
git clone <repository-url>
cd mocanet

# Create virtual environment and install dependencies
make setup

# Verify installation
make test
```

## 📁 Project Structure

```
mocanet/
├── src/mocanet/          # Core implementation
│   ├── layers.py         # TokenRouter, MemoryBank, ExpertLayer
│   ├── model.py          # Main MOCA-Net architecture
│   ├── data.py           # Data loaders and datasets
│   ├── train.py          # Training loop and trainer
│   ├── eval.py           # Evaluation and metrics
│   ├── ablation.py       # Ablation studies
│   └── utils.py          # Utility functions
├── configs/               # Configuration files
│   ├── base.yaml         # Base configuration
│   ├── copy_task.yaml    # Copy task settings
│   └── text_cls.yaml     # Text classification settings
├── tests/                 # Unit tests
├── scripts/               # Utility scripts
└── docs/                  # Documentation
```

## 🚀 Usage Examples

### Training

```bash
# Train on copy task (default)
make train

# Train on text classification
make demo

# Quick copy task run (1000 steps)
make run-copy

# Quick text classification run (500 steps)
make run-text
```

### Evaluation

```bash
# Evaluate trained model
python -m mocanet.eval runs/mocanet_best.pt --task copy

# Evaluate with custom config
python -m mocanet.eval runs/mocanet_best.pt --config configs/text_cls.yaml
```

### Ablation Studies

```bash
# Run all ablation studies
make ablate

# Results saved to runs/ablation/ablation_results.json
```

### Visualization

```bash
# Generate all plots
python scripts/plot_runs.py

# Generate specific plot types
python scripts/plot_runs.py --plot-type training
python scripts/plot_runs.py --plot-type ablation
python scripts/plot_runs.py --plot-type evaluation
```

## ⚙️ Configuration

### Model Architecture
```yaml
model:
  embedding_dim: 128      # Token embedding dimension
  num_experts: 4          # Number of expert networks
  num_memory_slots: 32    # Number of memory slots
  top_k_experts: 2        # Top-k experts to route to
  router_temperature: 1.0 # Router temperature for sparsity
  budget_loss_weight: 0.05 # Weight for budget loss term
```

### Training
```yaml
training:
  batch_size: 32          # Training batch size
  max_steps: 5000         # Maximum training steps
  learning_rate: 1e-3     # Learning rate
  warmup_steps: 200       # Warmup steps
  gradient_clip_norm: 1.0 # Gradient clipping
```

## 📈 Expected Results

### Copy Task
- **Target**: ≥95% accuracy on sequences ≤60 tokens
- **Runtime**: ≤10 minutes on CPU
- **Memory**: <4GB RAM usage

### Text Classification
- **Target**: ≥80% accuracy on synthetic SST-2 subset
- **Runtime**: ≤5 minutes on CPU
- **Dataset**: 10k synthetic samples

## 🔬 Ablation Studies

The framework supports several ablation studies:

1. **No Memory**: Disable external memory bank
2. **No Experts**: Use single expert instead of mixture
3. **Dense Routing**: Use all experts instead of sparse routing
4. **Smaller Model**: Reduce model size by half

Run with: `make ablate`

## 🧪 Testing

```bash
# Run all tests
make test

# Run specific test file
python -m pytest tests/test_layers.py -v

# Run with coverage
python -m pytest tests/ --cov=src/mocanet --cov-report=html
```

## 📊 Monitoring

Training progress is logged with:
- **Rich**: Beautiful console output with progress bars
- **TensorBoard**: Training curves and metrics
- **Checkpoints**: Model saves every 1000 steps

## 🚨 Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size in config
2. **Slow training**: Ensure `num_workers=0` for CPU training
3. **Import errors**: Run `make setup` to install dependencies

### Performance Tips

- Use `batch_size=16` for text classification on CPU
- Set `max_steps=1000` for quick experiments
- Enable `gradient_clip_norm=1.0` for stability

## 🔮 Future Work

1. **Learned Write Policy**: Adaptive memory update strategies
2. **KV Compression**: Efficient memory representation
3. **Hierarchical Experts**: Multi-level expert organization
4. **Retrieval-Augmented Tasks**: Integration with external knowledge
5. **Dynamic Routing**: Adaptive expert selection based on input complexity

## 📚 References

- **Mixture of Experts**: [Shazeer et al. (2017)](https://arxiv.org/abs/1701.06538)
- **External Memory**: [Graves et al. (2014)](https://arxiv.org/abs/1410.5401)
- **Sparse Routing**: [Lepikhin et al. (2020)](https://arxiv.org/abs/2006.16668)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass: `make test`
5. Submit a pull request

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- PyTorch team for the excellent deep learning framework
- Rich library for beautiful console output
- Hydra for configuration management
- The open-source ML community for inspiration

---

**Ready to explore efficient neural architectures? Start with `make setup` and dive into the world of MOCA-Net! 🚀**
