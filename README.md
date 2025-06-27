# LBCMamba

## 📌 Introduction

LBCMamba is a model based on the Mamba architecture, designed for sequence modeling tasks. The project supports using either the official implementation (`mamba_ssm`) or approximate version (which has already been integrated into the model).
---

## 🚀 Quick Start

### Execution Flow

By running the `main.py` script, you can perform the entire workflow, including:

- Model training  
- Model testing  
- Printing evaluation metrics  

### Experimental Record

- **Training logs**: Stored in `train.log`, containing detailed training information.  
- **Model weights**: Saved as `checkpoint_test.pth`, which can be used to reproduce test results.

---

## 📁 Dataset

You can download the required dataset from the following link:

🔗 [Dataset Link](https://www.cbr.washington.edu/dart/inventory)

Make sure to place the data in the correct directory before running training or testing.

---

## 🛠️ Dependencies

It is recommended to run this project in PyCharm or another Python development environment. Main dependencies include (install manually if not automatically handled):

```bash
pip install mamba-ssm  # For the official Mamba implementation
