# Interactive Inference Guide

This document provides comprehensive information about using MOCA-Net's interactive inference capabilities for sentiment analysis.

## Overview

MOCA-Net provides an interactive chat interface that allows you to input text and receive real-time sentiment predictions. The system is designed to be user-friendly and provides detailed insights into the model's predictions.

## Quick Start

### 1. Test Model Quality First

Before using interactive inference, always test your model quality:

```bash
make test-quality
```

This will:
- Compare different checkpoints
- Show accuracy on test sentences
- Recommend the best checkpoint to use
- Identify any training issues

### 2. Start Interactive Chat

```bash
make chat
```

This automatically uses the best available checkpoint (`mocanet_final.pt`).

### 3. Alternative: Direct Script Usage

```bash
python scripts/interactive_inference.py runs/mocanet_final.pt
```

## Interactive Features

### Real-time Sentiment Analysis
- Input any text and get instant predictions
- Support for sentences, phrases, and short paragraphs
- Automatic tokenization and preprocessing

### Confidence Scoring
- **High Confidence** (>0.8): Model is very certain about prediction
- **Medium Confidence** (0.6-0.8): Model is reasonably certain
- **Low Confidence** (<0.6): Model is uncertain, may need more training

### Rich Output Format
- **Predicted Sentiment**: Positive or Negative
- **Confidence Score**: How certain the model is
- **Probability Distribution**: Scores for both classes
- **Confidence Level**: Visual indicator (High/Medium/Low)

### Built-in Commands
- `help` - Show available commands
- `stats` - Display model architecture and configuration
- `quit` - Exit the session

## Model Checkpoints

### Recommended Checkpoints

| Checkpoint | Step | Validation Accuracy | Use Case |
|------------|------|-------------------|----------|
| `mocanet_final.pt` | 2000 | 96.4% | **Production inference** |
| `mocanet_step_1000.pt` | 1000 | ~90% | Development/testing |
| `mocanet_step_0.pt` | 0 | ~50% | **Avoid for inference** |

### Checkpoint Quality Indicators

- **File Size**: Larger files generally indicate more training steps
- **Validation Accuracy**: Should be >90% for good inference quality
- **Training Step**: Higher steps indicate more training completed

## Usage Examples

### Basic Sentiment Analysis

```
Enter text: This movie was absolutely fantastic!
Sentiment: Positive
Confidence: 0.892
Negative Probability: 0.108
Positive Probability: 0.892
Confidence Level: High
```

### Neutral/Complex Text

```
Enter text: I'm feeling quite neutral about this situation.
Sentiment: Positive
Confidence: 0.514
Negative Probability: 0.486
Positive Probability: 0.514
Confidence Level: Low
```

### Negative Sentiment

```
Enter text: The customer service was disappointing and unhelpful.
Sentiment: Negative
Confidence: 0.823
Negative Probability: 0.823
Positive Probability: 0.177
Confidence Level: High
```

## Troubleshooting

### Common Issues

#### 1. Poor Prediction Quality
**Symptoms**: Random predictions, low confidence, incorrect classifications

**Solutions**:
- Run `make test-quality` to verify checkpoint quality
- Ensure you're using `mocanet_final.pt` (not `mocanet_best.pt`)
- Check training logs for convergence issues
- Verify training completed successfully

#### 2. Low Confidence Scores
**Symptoms**: All predictions have confidence <0.6

**Solutions**:
- Model may need more training steps
- Check training configuration and hyperparameters
- Verify dataset quality and preprocessing
- Consider retraining with different parameters

#### 3. Import Errors
**Symptoms**: ModuleNotFoundError or ImportError

**Solutions**:
- Ensure virtual environment is activated: `. venv/bin/activate`
- Run `make setup` to install dependencies
- Check Python path and module structure
- Verify all required packages are installed

#### 4. Memory Issues
**Symptoms**: Out of memory errors during inference

**Solutions**:
- Use CPU inference instead of GPU
- Reduce batch size in configuration
- Close other applications to free memory
- Use smaller model configurations

### Performance Optimization

#### For Best Results
- Use the final checkpoint (`mocanet_final.pt`)
- Ensure sufficient training steps (≥2000 for SST-2)
- Use appropriate hyperparameters for your task
- Monitor training convergence

#### For Development
- Use intermediate checkpoints for testing
- Monitor validation accuracy during training
- Test on diverse sentence types
- Validate predictions against expected outcomes

## Advanced Usage

### Custom Checkpoints

To use a different checkpoint:

```bash
python scripts/interactive_inference.py path/to/your/checkpoint.pt
```

### Batch Processing

For processing multiple sentences programmatically:

```python
from scripts.interactive_inference import InteractiveInference

inference = InteractiveInference("runs/mocanet_final.pt")
sentences = ["Sentence 1", "Sentence 2", "Sentence 3"]

for sentence in sentences:
    prediction = inference.predict_sentiment(sentence)
    print(f"{sentence}: {prediction['predicted_label']}")
```

### Model Statistics

Access detailed model information:

```bash
# In interactive mode, type:
stats
```

This shows:
- Model architecture details
- Configuration parameters
- Vocabulary information
- Training statistics

## Best Practices

### Before Using Interactive Inference

1. **Verify Training Completion**: Ensure training finished successfully
2. **Test Model Quality**: Run `make test-quality` to verify performance
3. **Check Checkpoint**: Use the final or best validation checkpoint
4. **Validate Setup**: Ensure all dependencies are installed

### During Interactive Use

1. **Start Simple**: Test with clear positive/negative sentences first
2. **Monitor Confidence**: Pay attention to confidence scores
3. **Use Commands**: Utilize `help` and `stats` for information
4. **Test Edge Cases**: Try neutral or ambiguous sentences

### For Production Use

1. **Quality Assurance**: Thoroughly test before deployment
2. **Performance Monitoring**: Track prediction quality over time
3. **Error Handling**: Implement proper error handling for edge cases
4. **Documentation**: Maintain usage guidelines for end users

## Technical Details

### Model Architecture
- **Architecture**: MOCA-Net with mixture-of-experts and external memory
- **Input Processing**: Tokenization, embedding, and attention mechanisms
- **Output**: Binary classification with probability distributions
- **Memory**: External memory bank for long-term information retention

### Tokenization
- **Vocabulary**: Built from training data with configurable size
- **Unknown Tokens**: Handled with `<unk>` token
- **Padding**: Automatic padding to sequence length
- **Case**: Converted to lowercase for consistency

### Inference Pipeline
1. **Text Input**: User provides text string
2. **Tokenization**: Convert text to token IDs
3. **Model Forward Pass**: Process through MOCA-Net architecture
4. **Post-processing**: Apply softmax and extract predictions
5. **Output Formatting**: Present results in user-friendly format

## Support and Resources

### Getting Help
- **Documentation**: Check this guide and main README
- **Testing**: Use `make test-quality` for diagnostics
- **Issues**: Check troubleshooting section above
- **Development**: Review source code and configuration files

### Related Commands
- `make test-quality` - Test model quality
- `make demo-interactive` - Show interactive demo
- `make test-interactive` - Test interactive functionality
- `python scripts/test_model_quality.py` - Direct quality testing

### Configuration Files
- `configs/text_cls.yaml` - Text classification configuration
- `configs/base.yaml` - Base model configuration
- `scripts/interactive_inference.py` - Main interactive script
- `scripts/test_model_quality.py` - Quality testing script

---

**Note**: This interactive inference system is designed for research and development purposes. For production deployment, ensure proper testing, validation, and error handling are implemented.
