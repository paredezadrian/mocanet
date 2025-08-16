# MOCA-Net Documentation

Welcome to the MOCA-Net documentation! This guide provides an overview of all available documentation and how to navigate between different topics.

## Documentation Overview

### Core Architecture
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Detailed architecture design, mathematical foundations, and computational complexity
- **[DESIGN_DECISIONS.md](DESIGN_DECISIONS.md)** - Design rationale, architectural choices, and trade-off analysis

### Dataset Integration
- **[SST2_INTEGRATION.md](SST2_INTEGRATION.md)** - Stanford SST-2 dataset integration, implementation details, and performance benchmarks

### Interactive Features
- **[INTERACTIVE_INFERENCE.md](INTERACTIVE_INFERENCE.md)** - Comprehensive guide to interactive sentiment analysis, troubleshooting, and best practices

## Quick Navigation

### For New Users
1. Start with **[INTERACTIVE_INFERENCE.md](INTERACTIVE_INFERENCE.md)** to understand how to use the system
2. Review **[SST2_INTEGRATION.md](SST2_INTEGRATION.md)** to understand the dataset and performance
3. Explore **[ARCHITECTURE.md](ARCHITECTURE.md)** for technical details

### For Researchers
1. Begin with **[DESIGN_DECISIONS.md](DESIGN_DECISIONS.md)** to understand architectural choices
2. Study **[ARCHITECTURE.md](ARCHITECTURE.md)** for mathematical foundations
3. Review **[SST2_INTEGRATION.md](SST2_INTEGRATION.md)** for implementation insights

### For Developers
1. Check **[INTERACTIVE_INFERENCE.md](INTERACTIVE_INFERENCE.md)** for usage examples and troubleshooting
2. Review **[SST2_INTEGRATION.md](SST2_INTEGRATION.md)** for data handling patterns
3. Consult **[ARCHITECTURE.md](ARCHITECTURE.md)** for implementation details

## Key Features

### Interactive Inference System
- **Real-time sentiment analysis** with confidence scoring
- **Quality assurance** through model testing and checkpoint comparison
- **Rich terminal interface** with built-in commands and help system
- **Comprehensive troubleshooting** and best practices

### Architecture Highlights
- **Sparse mixture-of-experts** with dynamic routing
- **External memory bank** for long-term information retention
- **Budget-aware computation** with differentiable resource constraints
- **O(L) complexity** for efficient sequence modeling

### Dataset Integration
- **Stanford SST-2** real-world sentiment analysis data
- **Automatic vocabulary building** and tokenization
- **Performance benchmarks** and optimization tips
- **Troubleshooting guides** for common issues

## Getting Started

### 1. Setup and Installation
```bash
# Clone the repository
git clone https://github.com/paredezadrian/mocanet.git
cd mocanet

# Setup environment
make setup

# Verify installation
make test
```

### 2. Interactive Usage
```bash
# Test model quality first
make test-quality

# Start interactive chat
make chat

# Show demo
make demo-interactive
```

### 3. Training and Evaluation
```bash
# Train on SST-2 dataset
make demo

# Evaluate trained model
python -m mocanet.eval runs/mocanet_final.pt --task text_cls

# Generate plots
python scripts/plot_runs.py
```

## Documentation Structure

```
docs/
├── README.md                    # This overview file
├── ARCHITECTURE.md             # Technical architecture details
├── DESIGN_DECISIONS.md         # Design rationale and choices
├── SST2_INTEGRATION.md         # Dataset integration guide
└── INTERACTIVE_INFERENCE.md    # Interactive usage guide
```

## Contributing to Documentation

When updating documentation:

1. **Cross-reference** related documentation files
2. **Update this README.md** if adding new documentation
3. **Maintain consistency** in formatting and style
4. **Include examples** and usage patterns
5. **Add troubleshooting** sections for common issues

## Support and Resources

- **Main Repository**: [GitHub](https://github.com/paredezadrian/mocanet)
- **Issues**: Report bugs and request features on GitHub
- **Discussions**: Join community discussions for help and ideas
- **Code Examples**: Check the `scripts/` directory for usage examples

---

**Note**: This documentation is designed to be comprehensive and user-friendly. If you find any gaps or have suggestions for improvements, please contribute through GitHub issues or pull requests.
