# Model Checkpoints for BaMCo - Multimodal Knowledge Space

This directory is used to store pretrained model weights for the BaMCo - Multimodal Knowledge Space.

## Instructions

### 1. Download Model Weights

Visit the following Google Drive folder to download the pretrained model weights:

[BaMCo Knowledge Space Model Weights 🤗](https://drive.google.com/drive/folders/1uv7FsiafFWMQt8Se8hUOzg12qx4375H5?usp=sharing)

- Download the files named `<Dataset>_KnowledgeSpace.pt`.

### 2. Upload the Model File

Place the downloaded `<Dataset>_KnowledgeSpace.pt` file into this directory (here):

```
BaMCo/KSpace/src/checkpoints
```

### 3. Update Model Path in Code

Open your `main.py` file and update the model loading path to point to the new checkpoint location. For example:

```python
# main.py

 knowledge_encoder_checkpoint: Optional[str] = field(default="BaMCo/KSpace/src/checkpoints/<Dataset>_KnowledgeSpace.pt")
```

Make sure the path matches the location where you placed `<Dataset>_KnowledgeSpace.pt`.

---

**Note:**  
If you use a different filename or directory, update the path in your code accordingly.