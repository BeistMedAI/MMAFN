
# MMAFN: Multi-Modal Attention Fusion Network for ADHD Classification

This repository contains the implementation of the **"MMAFN: Multi-Modal Attention Fusion Network for ADHD Classification"** project. The model integrates multimodal data, including fMRI, sMRI, and phenotypic information, to enhance the classification accuracy for ADHD diagnosis.

## Table of Contents
1. [Project Structure](#project-structure)
2. [Algorithm Workflow](#algorithm-workflow)
3. [Environment Setup](#environment-setup)
4. [Training and Inference](#training-and-inference)
5. [Dataset Structure](#dataset-structure)
6. [Code Execution Example](#code-execution-example)
7. [Results](#results)
8. [Citation](#citation)

---

## 1. Project Structure
The directory structure of the project is as follows:

```
CBD2/
├── data/                      # Dataset processing module
│   └── dataset.py             # Script for dataset handling and preprocessing
├── Net/                       # Network architecture and model components
│   ├── api.py                 # API for model training and evaluation
│   ├── basicArchs.py          # Basic architectures for fMRI, sMRI, and text encoders
│   ├── cp_networks.py         # Cross-modal network architectures
│   ├── fusions.py             # Fusion strategies for multimodal data
│   ├── loss_functions.py      # Custom loss functions
│   ├── mamba_modules.py       # Mamba module for long-range dependency modeling
│   └── networks.py            # MMAFN model definition
├── utils/                     # Utility scripts for data processing and evaluation
│   ├── Clinical_data_processor.py  # Processes clinical text data
│   ├── CT_processer.py             # Handles CT image processing if applicable
│   ├── model_object.py             # Model object handler
│   └── observer.py                # Monitoring and logging
├── launch.py                  # Main script to launch training or evaluation
└── main.py                    # Core script to execute the MMAFN model pipeline
```

## 2. Algorithm Workflow
The MMAFN model is designed with the following main components:

1. **Feature Extraction**: Uses 3D ResNet-50 for fMRI and sMRI feature extraction, and BioBERT for encoding phenotypic text data.
2. **Multi-Modal Attention Fusion Module**: Cross-attention mechanism for data fusion, leveraging the Mamba module for improved dependency modeling.
3. **Classification**: Outputs ADHD diagnosis based on fused features using a DenseNet-based classifier.

A diagram explaining the architecture is shown below:

![MMAFN Architecture](path/to/your/architecture_image.png)

## 3. Environment Setup
This project requires Python 3.8 and the following dependencies:

- **PyTorch**: 2.0.0
- **CUDA**: 11.8 (for GPU support)

To set up the environment, you can use:

```bash
pip install -r requirements.txt
```

Alternatively, using `conda`:

```bash
conda create -n mmafn python=3.8
conda activate mmafn
conda install pytorch==2.0.0 cudatoolkit=11.8 -c pytorch
pip install -r requirements.txt
```

### Hardware Requirements
For optimal performance, we recommend using an NVIDIA GPU (such as A100) with at least 16GB VRAM.

## 4. Training and Inference
### Training
To train the MMAFN model, run the following command:

```bash
python launch.py --mode train --data_path ./data --epochs 300 --batch_size 2
```

- **Parameters**:
  - `--mode`: Set to `train` for training.
  - `--data_path`: Path to the dataset.
  - `--epochs`: Number of training epochs (default: 300).
  - `--batch_size`: Batch size for training (default: 2).

### Inference
To perform inference with a trained model, use:

```bash
python launch.py --mode test --model_path ./models/mmafn.pth --data_path ./data
```

- **Parameters**:
  - `--mode`: Set to `test` for inference.
  - `--model_path`: Path to the pre-trained model file.
  - `--data_path`: Path to the dataset for inference.

## 5. Dataset Structure
Organize the dataset in the following structure:

```
data/
├── fMRI/                # fMRI data
├── sMRI/                # sMRI data
└── text/                # Phenotypic data (e.g., age, gender, IQ, handedness)
```

Ensure that the dataset contains the ADHD-200 dataset with appropriate preprocessing for fMRI and sMRI data, and phenotypic information for text encoding.

### Data Preprocessing
- **MRI Data**: Standardize fMRI and sMRI images.
- **Phenotypic Data**: Encode using BioBERT to extract relevant features.

## 6. Code Execution Example
Here are some examples of how to execute the code:

### Training Example
```bash
python launch.py --mode train --data_path ./data --epochs 300 --batch_size 2
```

### Inference Example
```bash
python launch.py --mode test --model_path ./models/mmafn.pth --data_path ./data
```

## 7. Results
The MMAFN model outperforms other approaches in ADHD classification across several metrics.

| Method          | Modality             | Accuracy | Precision | Recall | F1 Score | AUC  |
|-----------------|----------------------|----------|-----------|--------|----------|------|
| MMAFN (Ours)    | fMRI+sMRI+Text       | 83.51%   | 88.39%    | 77.46% | 0.8255   | 0.8237 |

### Ablation Study
| Configuration          | Accuracy | Precision | Recall | F1 Score | AUC  |
|------------------------|----------|-----------|--------|----------|------|
| fMRI-only              | 78.24%   | 83.47%    | 65.23% | 0.7322   | 0.7648 |
| sMRI-only              | 74.15%   | 82.13%    | 58.41% | 0.6847   | 0.7415 |
| Text-only              | 69.88%   | 78.26%    | 47.89% | 0.5952   | 0.6832 |
| **MMAFN (All Modalities)** | **83.51%** | **88.39%** | **77.46%** | **0.8255** | **0.8237** |

## 8. Citation
If you use this code or find it helpful in your research, please cite our paper:

```plaintext
Jia, J., Liang, R., Zhang, C., et al. (2023). "MMAFN: Multi-Modal Attention Fusion Network for ADHD Classification." IEEE International Symposium on Biomedical Imaging (ISBI).
```

---

For additional details, please refer to our [paper](path/to/your/paper).
