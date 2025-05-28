# NYCU STinVRuDL 2025 Spring HW4
Student ID: 313561001   
Name: 林家輝

## Introduction
The goal of this project is to develop a high-performance **blind image restoration** system using the **PromptIR** framework. PromptIR is a **prompt-based, Transformer-driven** model designed for **all-in-one image restoration tasks** such as denoising, deblurring, deraining, and more. It leverages **hierarchical transformer encoders and decoders** along with **task-specific prompt generation** modules to conditionally guide restoration. 

Key strategies in this project include: 
* **Architecture customization** by experimenting with different Transformer block types (e.g., vanilla vs. Swin) and depth configurations.
* **Prompt injection** to enable adaptive, task-aware processing across different restoration objectives.
* **Mixed-precision distributed training** (using PyTorch Lightning DDP with AMP) for efficiency and scalability.
* **Quantitative evaluation** using **PSNR** and **SSIM** metrics to benchmark restoration quality across variants.

## 🧪 Experiment Setup 
- Architecture: PromptIR (4-stage hierarchical Transformer encoder-decoder)
- Backbone Variants:
  - PromptIR: Default architecture with vanilla Transformer blocks
  - PromptIR-T: More Transformer depth (6, 8, 8, 10) + refinement depth 8
  - PromptIR-S_LR: Swin Transformer used in all blocks except latent & refinement
  - PromptIR-S_enc: Swin Transformer used only in encoder blocks
  - PromptIR-S_enc-T: Combines Swin encoder + deep PromptIR-T structure -
- Swin Transformer Settings:
  - All Swin blocks use window_size = 8
- Data Augmentation:
  - Random crop
  - Horizontal flip
  - Vertical flip
- Training Setup:
  - Distributed training (DDP) with mixed precision (AMP)
  - Batch size: 8 per GPU - Optimizer: AdamW
  - Learning rate scheduler: Cosine Annealing
- Evaluation Metrics:
  - PSNR (Peak Signal-to-Noise Ratio)
  - SSIM (Structural Similarity Index)

## 📊 Results and Findings 

| Model Variant | PSNR | SSIM | 
|---------------------|-------|--------| 
| PromptIR | 29.72 | 0.9064 | 
| **PromptIR-T** | **30.26** | **0.9145** | 
| PromptIR-S_LR | 28.82 | 0.8925 | 
| PromptIR-S_enc | 29.95 | 0.9098 | 
| PromptIR-S_enc-T | 29.49 | 0.9028 | 

## 🛠️ How to Install and Run

### 1. Clone the Repository

```bash
git clone https://github.com/Zryxion/Selected-Topics-in-Visual-Recognition
cd Selected-Topics-in-Visual-Recognition/HW4
```

### 2. Create and Activate the Conda Environment

```bash
conda env create -f environment.yml
conda activate promptir
```

### 3. Train the Model (Example with Deraining and Dehazing)

```bash
python train.py \
  --epochs 150 \
  --de_type derain dehaze \
  --dehaze_dir ./path_to_data/ \
  --derain_dir ./path_to_data/ \
  --num_gpus 2 \
  --num_workers 12 \
  --batch_size 3
```

### 4. Run Evaluation on a Specific Task (e.g., Derain)

```bash
python test.py \
  --derain_path ./path_to_data/ \
  --ckpt_name ./path_to_checkpoint/epoch=XXX-step=YYYYY.ckpt \
  --cuda 0
```

### 5. Run Demo on Test Images

```bash
python demo.py \
  --test_path ./path_to_data/ \
  --ckpt_name ./path_to_checkpoint/epoch=XXX-step=YYYYY.ckpt
```

## Performance Snapshot

![](https://github.com/Zryxion/Selected-Topics-in-Visual-Recognition/blob/main/HW4/image/placement.png)

## Acknowledgments

This project is based on the [PromptIR: Prompting for All-in-One Blind Image Restoration (NeurIPS'23)](https://github.com/va1shn9v/PromptIR). 

