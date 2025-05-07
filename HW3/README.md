# NYCU STinVRuDL 2025 Spring HW3
Student ID: 313561001   
Name: 林家輝

## Introduction

The goal of this project is to develop a robust nucleus instance segmentation system using the **Mask R-CNN** architecture with a **ResNet-FPN** backbone. The model is trained to segment and classify nuclei across **four distinct types**, leveraging **transfer learning** with **COCO-pretrained weights** and a staged fine-tuning approach.
Key strategies include: 
* **Dropout regularization** in residual/identity blocks of ResNet to reduce overfitting.
* **Selective layer freezing/unfreezing** to control learning stability.
* **Data augmentation** via `imgaug` to improve generalization.
* **COCO-style evaluation metrics** to benchmark performance.

## Experiment Setup


- **Architecture**: Mask R-CNN + ResNet50-FPN or ResNet101-FPN backbone  
- **Pretrained Weights**:
  - ImageNet (for ResNet50)
  - COCO (for ResNet101)
- **Customizations**:
  - Dropout(0.1) in residual or identity blocks (depending on model variant)
- **Image Processing**:
  - Resize and center crop to 512×512
  - Enable mini-masks (56×56)
  - Multi-scale anchors: [16, 32, 64, 128, 256]
- **Data Augmentation (imgaug)**:
  - Horizontal and vertical flip (50%)
  - Rotations: 90°, 180°, 270°
  - Brightness adjustments
  - Gaussian blur
- **Training Setup**:
  - Stage 1: Train heads only (20 epochs)
  - Stage 2: Fine-tune full model (40 epochs)
  - Batch size: 2
- **Evaluation**:
  - COCO mAP@[0.50:0.95]
  - Validation loss (total, mask, bbox, RPN)
    
## Results and Findings

| Model Variant               | Pretrained | mAP@[0.50] | Val Loss |
|----------------------------|------------|------------|----------|
| Mask R-CNN (ResNet50)      | ImageNet   | 0.455      | 1.114    |
| ResNet50 + Dropout         | ImageNet   | **0.524**  | 1.231    |
| ResNet101 + Dropout*       | COCO       | 0.492      | **0.926**|

> `*` Dropout applied to end of residual blocks.  
> Dropout improves generalization, though may slightly raise training loss.

## 🔧 How to Install and Run

### 1. Clone the Repository

```bash
git clone https://github.com/Zryxion/Selected-Topics-in-Visual-Recognition
cd Selected-Topics-in-Visual-Recognition/HW3
```

### 2. Create a Conda Environment and Activate It

```bash
conda env create -f environment.yml
conda activate Mask_RCNN
```

## 🚀 How to Use

All functionality is accessed through a single entry point (`main.py`) using command-line arguments:

### Training

```bash
python main.py <command> --weights <path> [--dataset <path>] [--logs <path>] [--subset <name>] --type <1|2>
```

Optional:
* `<command>`: `train` or `detect`
* `--weights`: Path to weights `.h5` file or `'coco'`, `'last'`, `'imagenet'`
* `--dataset`: Path to the dataset (required for training)
* `--logs`: Directory to save logs and checkpoints (default: `logs/`)
* `--subset`: Subset of the dataset to run detection on (required for detection)
* `--type`:
  * `1`: Use standard Mask R-CNN
  * `2`: Use variant with dropout in the identity block



### Evaluation (on validation set)

```bash
python check.py
```

### Testing (on unseen image folder)

```bash
python submission.py
```

> Both **check.p**y and **submission.py** uses the last weight.  
> Path to weight other than the last should be initialized in the code.

## Performance Snapshot

![](https://github.com/Zryxion/Selected-Topics-in-Visual-Recognition/blob/main/HW3/image/placement.png)
