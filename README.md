# CSC3831: Data Engineering, Machine Learning & Computer Vision Coursework

## Project Overview

This repository contains the full workflow for the CSC3831 coursework, spanning from initial data preprocessing to advanced deep learning model implementation. The project is divided into three primary tasks:

1. **Data Engineering**: Handling missing data and feature engineering on housing datasets.
2. **Machine Learning**: Dimensionality reduction and denoising using the MNIST digit dataset.
3. **Computer Vision**: Building and analyzing a custom Convolutional Neural Network (CNN) for image classification.

---

## Task 1: Data Engineering (`part_1.ipynb`)

**Objective:** Address data quality issues in a corrupted housing dataset (Missing at Random - MAR) and predict median house values.

* **Key Processes:**
* **Data Cleaning:** Removal of artifacts and redundant features.
* **Imputation Techniques:** Evaluated **KNN Imputation** vs. **MICE (Multivariate Imputation by Chained Equations)**.
* **Analysis:** Comparison of RMSE scores and Kernel Density Estimate (KDE) plots to determine which model best preserves the "natural" distribution of the data.


* **Findings:** MICE generally produced better results for machine learning models (lower RMSE), though it risked artificially inflating density in certain linear distributions.



---

## Task 2: Machine Learning (`part_2.ipynb`)

**Objective:** Explore image reconstruction and denoising using Principal Component Analysis (PCA) on the MNIST dataset.

* **Key Processes:**
* **Dimensionality Reduction:** Utilized PCA to find the minimum number of components required for visualization.
* **Denoising:** Applied PCA-based reconstruction to remove Gaussian noise from digit images.


* **Findings:** A **65% threshold** for explained variance was identified as the strong contender for dimensionality reduction; lower thresholds resulted in unrecognizable, blurry images.



---

## Task 3: Computer Vision (`part_3.ipynb` & `Task3_Report.pdf`)

**Objective:** Design and train a custom CNN to classify the CIFAR-10 dataset (specifically focusing on ship detection) and interpret internal feature maps.

### Model Architecture

The network is a "Simple CNN" inspired by the **VGG model**, utilizing stacked  convolutional filters to capture complex features with fewer parameters.

| Layer Type | Configuration | Output Map |
| --- | --- | --- |
| **Input** | 3-Channel RGB |  |
| **Block 1** |  Conv2D (32/64) + MaxPool |  |
| **Block 2** |  Conv2D (64/128) + MaxPool |  |
| **Block 3** |  Conv2D (128/256) + MaxPool |  |
| **Classifier** | Flatten + 2 Linear Layers + ReLU | 10 Classes |

### Training Strategy

* **Optimizer:** Adam Optimizer.
* **Regularization:** Early stopping based on validation loss to prevent overfitting.


* **Feature Visualization:** Analysis of the "Concept Detection" phase (Block 3), where  maps provide high-level summaries (e.g., distinguishing ship body vs. water).



---

## How to Use

1. **Prerequisites:** Python 3.12, PyTorch, Keras, Pandas, Seaborn, and Matplotlib.
2. **Execution:**
* Run `part_1.ipynb` for data imputation analysis.
* Run `part_2.ipynb` to view PCA denoising results.
* Run `part_3.ipynb` to train the CNN and generate feature map visualizations.


3. **Report:** For a detailed breakdown of the CNN design decisions and literature references (e.g., Simonyan & Zisserman), refer to `Task3_Report.pdf`.



## Author

**Sanjana Muppasani**