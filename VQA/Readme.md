# BaMCo - Knowledge DrivenVisual Question Answering

This folder contains the source code for the BaMCo - Visual Question Answering.

This repository contains the codebase for BaMCo VQA, a multimodal knowledge-driven biomedical Visual Question Answering (VQA). The project leverages large language models, vision encoders, and knowledge graph integration for advanced VQA tasks on biomedical datasets.

## Folder Structure

```
src/
├── main.py                       # Main training and evaluation script
├── dataset/
│   ├── class_embeddings_*.npy    # Precomputed class embeddings
│   ├── dataset_info.py           # Dataset metadata and helpers
│   └── multi_dataset.py          # Dataset loaders for VQA tasks
├── model/
│   ├── __init__.py
│   ├── BaMCo_VQA_arch.py         # Model architecture definitions
│   ├── loss.py                   # Loss functions (e.g., BCELoss)
│   ├── cache/                    # Model cache
│   ├── language_model/           # Language model wrappers (GPT2, Llama, etc.)
│   ├── multimodal_encoder/       # Vision and knowledge encoder modules
│   └── multimodal_projector/     # Projector modules for multimodal fusion
├── outputs/                      # Model outputs and checkpoints
├── train/
│   └── BaMCo_VQA_trainer.py      # Custom Trainer class
├── utils/
│   └── dist_utils.py             # Distributed training utilities
└── .gitignore
```

## Features

- **Multimodal VQA**: Integrates vision, language, and knowledge graph information.
- **Flexible Model Architectures**: Supports Llama, GPT2, and custom adapters.
- **Custom Loss Functions**: Includes BCELoss and others for VQA tasks.
- **Dataset Support**: PathVQA, Slake, VQARAD, and more.
- **Trainer Integration**: Uses Hugging Face Trainer with custom hooks.
- **Evaluation Metrics**: BLEU, ROUGE, METEOR, BERTScore, and accuracy.

## Getting Started

### Requirements

- Python 3.8+
- PyTorch
- Transformers
- scispaCy
- datasets
- wandb (optional)
- tqdm
- PIL
- numpy

Install dependencies:
```bash
pip install -r requirements.txt
```

### Training & Evaluation

1. **Configure your dataset and model arguments** in `main.py` or via command-line.
2. **Run training:**
    ```bash
    python src/main.py
    ```
3. **Evaluation** is performed automatically after training, or you can set `eval_only=True` in the arguments.

### Checkpoints & Outputs

- Model checkpoints and logs are saved in the `outputs/` directory.
- Evaluation results are saved as JSON files.