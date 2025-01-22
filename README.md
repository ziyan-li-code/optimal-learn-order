Codes for the continual learning project with optimal order model

## Citation
Thank you for reading and using this code! If you find it useful and use it in your research or work, please cite the associated article:

[Author Name(s)], "[Title of Your Article]," *Journal/Conference Name*, Year, DOI: [Insert DOI link here].

### BibTeX
You can use the following BibTeX entry: \
@article{YourCitationKey, \
  author    = {Your Name and Co-author Name}, \
  title     = {Title of Your Article}, \
  journal   = {Journal/Conference Name}, \
  year      = {Year}, \
  volume    = {Volume}, \
  number    = {Issue}, \
  pages     = {Page range}, \
  doi       = {Insert DOI link here}, \
  url       = {Insert URL here} \
}

## Explanations for Modules

This repository contains several modules designed for machine learning tasks. Below is an explanation of each module and its purpose:

### `continual_model.py`
This module provides tools for model training, including implementations for **continual learning**. It supports tasks such as incremental learning with multiple datasets.

### `dataset_process.py`
This module handles the **preprocessing** and **uploading** of TensorFlow datasets. Examples of supported datasets include:
- CIFAR-10
- CIFAR-100
- Fashion MNIST

### `file_extract.py`
This module extracts data from `.csv` files. It processes files containing **labels** and **accuracies** to prepare them for further analysis.

### `group_split.py`
This module allows for selecting specific classes from a dataset. It splits the selected classes into **task-specific groups** for targeted training.

### `network.py`
This module builds neural networks using **Flax** (`flax.nn`). It supports:
- **Convolutional Neural Networks (CNNs)**
- **Nonlinear Neural Networks**

### `opt_order.py`
This module computes the **optimal order** for tasks or datasets based on the **Hamiltonian Path** or **Periphery-Core Model**.

### `similarity.py`
This module calculates **inter-task similarity** using techniques such as:
- **Zero-shot learning**
- **gHg (gradient-Hessian-gradient) analysis**

---

## Example Repository
The `examples` directory contains experiment examples with detailed instructions in the respective `README.md` file.

## General Parameters

The following parameters are used in this machine learning model:

| Parameter Name     | Description                                   | Default Value |
|--------------------|-----------------------------------------------|---------------|
| `learning_rate`    | Learning rate for the optimizer.             | `0.001`       |
| `batch_size`       | Number of samples per batch during training. | `4`        |
| `ghg_batch_size`       | Number of samples per batch during gradient-Hessian-gradient  similarity calculation. | `256`        |
| `num_epochs`       | Number of training epochs.                   | `5`         |
| `optimizer`        | Optimizer used for training.                 | `Adam`        |
| `loss_function`    | Loss function used to train the model.       | `CrossEntropy`|
| `shuffle_size`     | Size of the buffer used for shuffling the dataset.| `1000`       |

