# BaMCo - Multimodal Knowledge Space Pretraining

This folder contains the source code for the BaMCo - Multimodal Knowledge Space Pretraining.

Once the multimodal encoders have optimized a specialized knowledge space, this knowledge is integrated into the VQA model to enhance its question answering performance.

## Folder Structure

```
src/
├── dataload.py            # Dataset loading and preprocessing
├── distributed.py         # Distributed training utilities
├── GLIMS.py               # GLIMS model or related utilities
├── loss.py                # Loss functions
├── main.py                # Main training and evaluation script
├── model.py               # Model definitions
├── params.py              # Argument parsing and default parameters
├── randaugment.py         # Data augmentation utilities
├── scheduler.py           # Learning rate scheduler
├── train.py               # Training and evaluation loops
├── Modules/               # Model submodules
│   ├── conv_generator.py
│   ├── CSAB.py
│   └── DepthWiseAtriousBlocks.py
└── .gitignore
```

### Requirements

- Python 3.8+
- PyTorch
- torchvision
- numpy
- transformers
- wandb (optional, for experiment tracking)
- tqdm
- opencv-python
- PIL

### Training

The dataset and the training parameters can be customized from the `params.py` script.

### Data

The code expects knowledge graph data in JSON format. Update the paths in the `params.py` script as needed.
