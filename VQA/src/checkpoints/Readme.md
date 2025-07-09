# Model Checkpoints for BaMCo VQA

This directory is used to store pretrained model weights for the BaMCo VQA project.

## Instructions

### 1. Download Model Weights

Visit the following Hugging Face collection to download the pretrained model weights:

[BaMCo Model Collection 🤗](https://huggingface.co/collections/yaziciz/bamco-686e27c7a6e410dbe6059010)

- Download the file named `pytorch_model_best.bin` from the appropriate model card.

### 2. Upload the Model File

Place the downloaded `pytorch_model_best.bin` file into this directory (here):

```
BaMCo/VQA/src/checkpoints/
```

### 3. Update Model Path in Code

Open your `main.py` file and update the model loading path to point to the new checkpoint location. For example:

```python
# main.py

checkpoint_dir: str = "BaMCo/VQA/src/checkpoints/pytorch_model_best.bin"
```

Make sure the path matches the location where you placed `pytorch_model_best.bin`.

---

**Note:**  
If you use a different filename or directory, update the path in your code accordingly.