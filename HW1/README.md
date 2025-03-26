# NYCU Computer Vision 2025 Spring HW1
Student ID: 313561001   
Name: 林家輝

## Introduction

The primary objective of this project is to develop an image classification model utilizing the 
ResNeXt architecture. The goal is to train the model to recognize and classify images into **100 
different classes** with high accuracy. The core idea of my method is to use **Resnext50(32x4d)** and leverage transfer learning 
from pre-trained ImageNet weights, optimizing it for the custom dataset through techniques such as 
layer freezing, weighted loss functions, data augmentation and bagging.

## How to Install

### 1. Clone the Repository
Clone the project with:
``` 
git clone https://github.com/Zryxion/Selected-Topics-in-Visual-Recognition
cd Selected-Topics-in-Visual-Recognition/HW1
```

### 2. Create a Conda Environment and Activate
Create a new Conda environment for the project:
``` 
conda create --name my_env python=3.10
conda activate my_env
```

### 3. Install Dependencies
Install the required packages from requirements.txt:
``` 
pip install -r requirements.txt
```
### 4. Start Training and Testing
You can now start Training and Testing by:
``` 
python train.py
python test.py
```
***_Remember to change the directory variable inside the files to your local directory_***

### 5. Plot Learning Curve (Optional)
You can plot the learning curve by running:
``` 
python test.py
``` 
***_Only run when you have the pickle files_***
## Performance Snapshot 🚀
![](https://github.com/Zryxion/Selected-Topics-in-Visual-Recognition/blob/main/HW1/leaderboard.png)
