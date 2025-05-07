# NYCU STinVRuDL 2025 Spring HW3
Student ID: 313561001   
Name: 林家輝

## Introduction

The goal of this project is to develop a robust **object detection system** based on the **Faster R-CNN** architecture with a **ResNet50-FPN** backbone, capable of detecting and classifying **10 numeric digits**. The approach leverages **transfer learning** from COCO-pretrained weights, with selective fine-tuning and various regularization strategies to optimize model performance on a custom dataset.

To improve generalization and prevent overfitting, we introduce **dropout regularization** in the classification head and selectively unfreeze deeper layers (e.g., `layer4`) in the backbone. The model is trained and evaluated using **COCO-style mAP metrics** and **validation accuracy**, with training optimized via **mixed precision (AMP)** and **Distributed Data Parallel (DDP)**.

## Experiment Setup

- **Architecture**: Faster R-CNN + ResNet50-FPN backbone (pretrained on COCO)
- **Customizations**:
  - Dropout(0.3) in the classification head
  - Selective unfreezing of `layer4` and detection heads
- **Training Setup**:
  - Optimizer: SGD with momentum & weight decay
  - Mixed precision (AMP) and DDP for faster training
  - Evaluation via COCO mAP@[0.50:0.95] and validation accuracy

## Results and Findings

| Model Variant                   | mAP@[0.50:0.95] | Accuracy |
|-------------------------------|------------------|----------|
| Faster R-CNN (Base)           | 0.454            | 92.6%    |
| Faster R-CNN + Dropout        | 0.457            | 92.5%    |
| Faster R-CNN + Dropout + Ft   | **0.465**        | **93.6%**|

- **Dropout alone** leads to a small but consistent **mAP gain**, suggesting improved **bounding box localization** through regularization.
- However, **training loss curves** show **higher classification and box regression losses** for the dropout model, indicating slower convergence due to the stochastic nature of dropout.
- Despite this, **validation performance** improves slightly, implying **better generalization**.
- The best results are achieved when **dropout is combined with fine-tuning** of the deeper `layer4`, yielding the highest accuracy and mAP.

## 🔧 How to Install and Run

### 1. Clone the Repository

```bash
git clone https://github.com/Zryxion/Selected-Topics-in-Visual-Recognition
cd Selected-Topics-in-Visual-Recognition/HW1
```

### 2. Create a Conda Environment and Activate It

```bash
conda create --name my_env python=3.10
conda activate my_env
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 How to Use

All functionality is accessed through a single entry point (`main.py`) using command-line arguments:

### Training

```bash
python main.py --mode train --model-type 0
```

Optional:
- `--chkt-flag True`: Resume from checkpoint
- `--model-path`: Path to load model (default: `fasterrcnn_ddp.pth`)
- `--model-type`: Use `0` for baseline, `1` for Dropout version

### Evaluation (on validation set)

```bash
python main.py --mode eval --model-type 0
```

Optional:
- `--model-path`: Path to load model (default: `fasterrcnn_ddp.pth`)
- `--model-type`: Use `0` for baseline, `1` for Dropout version

### Testing (on unseen image folder)

```bash
python main.py --mode test --model-type 0
```
Optional:
- `--model-path`: Path to load model (default: `fasterrcnn_ddp.pth`)
- `--model-type`: Use `0` for baseline, `1` for Dropout version
