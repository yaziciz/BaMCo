# 🧠 BaMCo: Balanced Multimodal Contrastive Learning for Knowledge-Driven Medical VQA

Welcome to **BaMCo**, a novel framework for multimodal, knowledge-driven biomedical Visual Question Answering. This repository contains the implementation of the paper, BaMCo, accepted to MICCAI 2025.

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/yaziciz/BaMCo.git
cd BaMCo
```

### 2. Install Requirements

```bash
conda env create -f environment.yml
conda activate bamco
```

### 3. Prepare Datasetsß

- Place your datasets under the appropriate folders in `KSpace/Datasets/` or use the predefined datasets, Slake, PathVQA and VQA-RAD.

### 4. Download Model Weights

- **VQA Model:**  
  Download `pytorch_model_best.bin` from  
  [Hugging Face BaMCo Collection](https://huggingface.co/collections/yaziciz/bamco-686e27c7a6e410dbe6059010)  
  and place it in `VQA/src/checkpoints/`.

- **Knowledge Encoder:**  
  Download `<Dataset>_KnowledgeSpace.pt` from  
  [Google Drive Knowledge Space Weights](https://drive.google.com/drive/folders/1uv7FsiafFWMQt8Se8hUOzg12qx4375H5?usp=sharing)  
  and place it in `KSpace/src/checkpoints/`.

### 5. Update Model Paths

- Edit `main.py` in both `VQA/src/` and `KSpace/src/` to point to the correct checkpoint files as described in the respective `Readme.md` files in each `checkpoints/` directory.

---

## 🏗️ Main Components

- **KSpace:**  
  Scripts for constructing and encoding biomedical knowledge sources.

- **VQA:**  
  End-to-end VQA pipeline, including data loading, model training, evaluation, and inference.

- **Checkpoints:**  
  Store and manage pretrained model weights for both knowledge encoders and VQA models.

---

## 📝 Citation

We appreciate your interest! If you use or refer to BaMCo in your research, please cite us:
The citation will be updated soon!

```bibtex
@inproceedings{BaMCo_MICCAI2025,
  title     = {BaMCo: Balanced Multimodal Contrastive Learning for Knowledge-Driven Medical VQA},
  author    = {Ziya Ata Yazici and Hazım Kemal Ekenel},
  booktitle = {International Conference on Medical Image Computing and Computer-Assisted Intervention},
  year      = {2025}
}
```

---

## 📬 Contact

For questions, issues, or contributions, please open an issue or pull request on GitHub.

---