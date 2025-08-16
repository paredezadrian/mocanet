#!/usr/bin/env python3
"""Demo script showing interactive inference capabilities of MOCA-Net SST-2 model."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from scripts.interactive_inference import InteractiveInference


def demo_interactive_inference():
    """Demonstrate interactive inference with example sentences."""
    print("MOCA-Net SST-2 Interactive Inference Demo")
    print("=" * 50)
    print()
    
    # Example sentences for demonstration
    example_sentences = [
        "This movie was absolutely fantastic and I loved every minute of it!",
        "The food was terrible and the service was even worse.",
        "The book was okay, nothing special but not bad either.",
        "I'm so excited about this amazing opportunity!",
        "This is the worst experience I've ever had.",
        "The weather is beautiful today and the flowers are blooming.",
        "I'm feeling quite neutral about this situation.",
        "This product exceeded all my expectations!",
        "The customer service was disappointing and unhelpful.",
        "What a wonderful day for a picnic in the park!"
    ]
    
    print("Example sentences for sentiment analysis:")
    for i, sentence in enumerate(example_sentences, 1):
        print(f"{i:2d}. {sentence}")
    
    print()
    print("To try interactive inference yourself, run:")
    print("  make chat")
    print("  or")
    print("  python scripts/interactive_inference.py runs/mocanet_best.pt")
    print()
    print("Available commands:")
    print("  - Type any sentence to analyze sentiment")
    print("  - Type 'help' for available commands")
    print("  - Type 'stats' to see model information")
    print("  - Type 'quit' to exit")
    print()
    print("The model will provide:")
    print("  - Predicted sentiment (Positive/Negative)")
    print("  - Confidence score")
    print("  - Probability for each class")
    print("  - Confidence level indicator")


if __name__ == "__main__":
    demo_interactive_inference()
