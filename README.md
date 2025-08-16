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

# Chat interactively with trained SST-2 model
make chat

# Show interactive inference demo
make demo-interactive
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

For a comprehensive understanding of the mathematical foundations, refer to [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). For detailed information about the Stanford SST-2 dataset integration, see [docs/SST2_INTEGRATION.md](docs/SST2_INTEGRATION.md). For comprehensive interactive inference usage, see [docs/INTERACTIVE_INFERENCE.md](docs/INTERACTIVE_INFERENCE.md).

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

# Train on text classification task (real SST-2 dataset)
make demo

# Quick training runs for rapid experimentation
make run-copy    # 1000 steps on copy task
make run-text    # 500 steps on text classification

# Test and debug SST-2 dataset integration
make test-sst2   # Test SST-2 dataset loading
make debug-sst2  # Debug data and model outputs

# Interactive inference with trained model
make chat        # Chat with trained SST-2 model (uses final checkpoint)
make demo-interactive  # Show interactive inference demo
make test-quality     # Test model quality and compare checkpoints
```

### Model Evaluation

Evaluate your trained models using the built-in evaluation framework:

```bash
# Evaluate trained model on copy task
python -m mocanet.eval runs/mocanet_best.pt --task copy

# Generate comprehensive training plots
python scripts/plot_runs.py
```

### Interactive Inference

Chat interactively with your trained SST-2 sentiment analysis model:

```bash
# Start interactive chat session (recommended)
make chat

# Test model quality first
make test-quality

# Show interactive demo
make demo-interactive

# Or run directly with custom checkpoint
python scripts/interactive_inference.py runs/mocanet_final.pt
```

**Interactive Features:**
- **Real-time sentiment analysis** of any text input
- **Confidence scores** and probability distributions
- **Model statistics** and configuration details
- **Rich terminal interface** with colored output
- **Built-in commands**: `help`, `stats`, `quit`

**Important Notes:**
- **Use `runs/mocanet_final.pt`** for the fully trained model (96.4% validation accuracy, step 2000)
- **Avoid `runs/mocanet_best.pt`** for inference (step 0, poor quality)
- **Test model quality** with `make test-quality` before using interactively

**Example Usage:**
```
Enter text: This movie was absolutely fantastic!
Sentiment: Positive
Confidence: 0.892
Negative Probability: 0.108
Positive Probability: 0.892
Confidence Level: High
```

**Interactive Workflow:**
1. **Test Model Quality**: Run `make test-quality` to verify checkpoint performance
2. **Start Interactive Session**: Use `make chat` to begin sentiment analysis
3. **Input Text**: Type any sentence to analyze sentiment
4. **View Results**: See prediction, confidence, and probability distributions
5. **Use Commands**: Type `help`, `stats`, or `quit` for additional features

**Available Commands:**
- `help` - Show available commands and usage
- `stats` - Display model architecture and configuration details
- `quit` - Exit the interactive session

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
- **text_cls.yaml**: Configuration for text classification tasks with real Stanford SST-2 dataset

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

# SST-2 Dataset Configuration
text_cls:
  use_real_sst2: true       # Use real Stanford SST-2 dataset
  dataset: "sst2"           # Full Stanford SST-2 dataset
  min_freq: 2               # Minimum token frequency for vocabulary
  max_vocab_size: 10000     # Maximum vocabulary size
```

## Expected Results

### Copy Task Performance
- **Target Accuracy**: ≥95% on sequences up to 60 tokens
- **Expected Runtime**: ≤10 minutes on CPU
- **Memory Usage**: <4GB RAM

### Checkpoint Information
- **`mocanet_final.pt`**: Final trained model (step 2000, 96.4% validation accuracy) - **Use for inference**
- **`mocanet_best.pt`**: Best validation checkpoint (step 0) - **Avoid for inference**
- **`mocanet_step_*.pt`**: Intermediate training checkpoints for analysis

### Text Classification Performance
- **Target Accuracy**: ≥95% on Stanford SST-2 dataset
- **Expected Runtime**: ≤10 minutes on CPU for 500 steps, ~6.5 minutes for 2000 steps
- **Dataset Size**: 67,349 training samples, 872 validation samples, 1,821 test samples
- **Real Dataset**: Full Stanford Sentiment Treebank v2 (SST-2) from Hugging Face

### Interactive Inference Capabilities
- **Real-time Analysis**: Instant sentiment predictions for any text input
- **Confidence Scoring**: Probability distributions and confidence levels (>0.8 for trained model)
- **User Interface**: Rich terminal-based interactive chat system with colored output
- **Model Insights**: Access to model statistics and configuration details
- **Quality Assurance**: Built-in model quality testing and checkpoint comparison
- **Performance**: 66.7% accuracy on test sentences with 96.4% validation accuracy

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

# Test interactive inference functionality
make test-interactive

# Test model quality and compare checkpoints
make test-quality

# Run specific test file with verbose output
python -m pytest tests/test_layers.py -v

# Generate coverage report
python -m pytest tests/ --cov=src/mocanet --cov-report=html
```

## Available Scripts

MOCA-Net provides several utility scripts for different purposes:

### Training and Evaluation
- **`scripts/plot_runs.py`**: Generate comprehensive training plots and visualizations
- **`scripts/debug_data.py`**: Debug data loading and preprocessing issues
- **`scripts/test_sst2.py`**: Test SST-2 dataset integration and loading

### Interactive Inference
- **`scripts/interactive_inference.py`**: Main interactive chat interface for sentiment analysis
- **`scripts/demo_interactive.py`**: Demo script showing interactive capabilities
- **`scripts/test_model_quality.py`**: Test and compare different model checkpoints

### Documentation
- **`docs/INTERACTIVE_INFERENCE.md`**: Comprehensive guide to interactive inference
- **`docs/ARCHITECTURE.md`**: Detailed architecture and mathematical foundations
- **`docs/SST2_INTEGRATION.md`**: SST-2 dataset integration details

### Usage Examples
```bash
# Test model quality before interactive use
make test-quality

# Start interactive chat
make chat

# Show interactive demo
make demo-interactive

# Generate training plots
python scripts/plot_runs.py
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
4. **Poor Interactive Inference Results**: Use `make test-quality` to verify model quality

### Interactive Inference Issues

1. **Random/Incorrect Predictions**: 
   - Ensure you're using `runs/mocanet_final.pt` (not `mocanet_best.pt`)
   - Run `make test-quality` to verify checkpoint quality
   - Check that training completed successfully (validation accuracy >90%)

2. **Low Confidence Scores**:
   - Model may need more training steps
   - Verify training configuration and hyperparameters
   - Check training logs for convergence

3. **Import Errors in Interactive Mode**:
   - Ensure virtual environment is activated: `. venv/bin/activate`
   - Run `make setup` to install all dependencies
   - Check Python path and module imports

### Performance Optimization Tips

- Use `batch_size=16` for text classification tasks on CPU
- Set `max_steps=1000` for rapid experimentation cycles
- Enable `gradient_clip_norm=1.0` for training stability
- For interactive inference, use the final checkpoint (step 2000) for best results

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
