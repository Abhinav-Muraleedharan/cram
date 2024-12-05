# Cumulative Retention Autoregressive Models (CRAM)

This repository contains the implementation of Cumulative Retention Autoregressive Models (CRAM), a novel architecture designed to address the limitations of attention-based Transformer architectures while maintaining linear computational complexity with respect to sequence length.

## Features

- Linear time complexity O(N) with respect to sequence length
- Nonlinear transformations for complex sequence modeling
- Efficient parallel training implementation in JAX/Flax
- Comprehensive benchmarking suite
- Easy-to-use training and evaluation scripts

## Installation

```bash
git clone https://github.com/yourusername/cram.git
cd cram
pip install -e .
```

## Getting Started

### Training a Model

```bash
python examples/train_cram.py \
    --config configs/base_config.py \
    --dataset wikitext-103-raw-v1 \
    --output_dir outputs/base_model
```

### Evaluation

```bash
python examples/evaluate_model.py \
    --model_path outputs/base_model \
    --dataset wikitext-103-raw-v1 \
    --split test
```

## Project Structure

```
cram/
├── configs/          # Configuration files
├── cram/            # Main package
│   ├── models/      # Model implementations
│   ├── modules/     # Core building blocks
│   ├── data/        # Dataset and dataloader utilities
│   ├── training/    # Training utilities
│   └── utils/       # Helper functions
├── tests/           # Unit tests
├── benchmarks/      # Benchmarking scripts
└── examples/        # Example usage scripts
```

## Benchmarks

Our model has been evaluated on various benchmarks:

1. Language Modeling
   - WikiText-103
   - The Pile
   - C4

2. Long-Range Understanding
   - Path-X task
   - Long Range Arena

## Contributing

We welcome contributions! Please read our [Contributing Guidelines](CONTRIBUTING.md) for details on how to submit pull requests.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{your_paper,
    title={Cumulative Retention Autoregressive Models},
    author={Your Name},
    journal={arXiv preprint arXiv:XXXX.XXXXX},
    year={2024}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.