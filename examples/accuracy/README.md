# Continual Learning Experiment Configuration

This repository contains machine learning experiments designed to evaluate the **accuracy of a series of continual learning tasks**. Below are descriptions of the key parameters used to configure the model, training process, and experimental setup.

## Parameters for Model Selection
- **`ds_type`**: Dataset type. Options include:
  - `'fashion_mnist'`: A grayscale dataset for image classification.
  - `'cifar10'`: A colored dataset with 10 classes.
  - `'cifar100'`: A colored dataset with 100 classes.
- **`nn_type`**: Neural network architecture type. Options include:
  - `'cnn2'`, `'cnn5'`: Convolutional neural networks with 2 or 5 layers.
  - `'nonlinear2'`, `'nonlinear5'`: Nonlinear models with 2 or 5 layers.
- **`sim_type`**: Similarity calculation model type used to measure transfer learning performance. Options:
  - `'zero_shot'`: Zero-shot learning evaluation.
  - `'ghg'`: Gradient-based similarity measurement.

## Parameters for Training Process
- **`num_task`**: The total number of tasks in the continual learning experiment (default: 5).
- **`num_output_classes`**: The number of output classes per task (default: 2).
- **`num_all_classes`**: The total number of classes in the dataset (e.g., 10 for CIFAR-10, 100 for CIFAR-100).
- **`learning_rate`**: Learning rate for model optimization (default: 0.001).
- **`num_regular_epochs`**: Number of epochs per task during the regular training phase (default: 5).
- **`num_continue_epochs`**: Number of epochs per task during the continued training phase (default: 5).
- **`batch_size`**: Batch size for training (default: 4).
- **`shuffle_size`**: Shuffle buffer size for loading data (default: 1000).
- **`image_size`**: Dimensions of input images:
  - `[28, 28, 1]` for grayscale datasets.
  - `[32, 32, 3]` for colored datasets.

## Parameters for Experiment Settings
- **`num_pick`**: Number of ways to randomly pick and group `num_task * num_class` classes from the total `num_all_classes` (default: 10).
- **`num_perm`**: Number of multi-permutations for each sample point, often used to analyze task ordering effects:
  - Options include 6, 30, or 50 for P = 3, 5, or 7.
- **`num_index`**: Job index for managing class splits and label splits (default: 1).
- **`ini_seed`**: Seed for initializing model parameters, ensuring reproducibility (default: 0).

## Purpose
This code is specifically designed to evaluate the performance of models in **continual learning scenarios** by:
- Training a single model across multiple sequential tasks.
- Measuring task-specific and overall accuracy across the sequence.
- Investigating task interference and transfer learning between tasks.

The parameters allow flexible configuration of dataset, model architecture, and training processes to suit various continual learning scenarios.

For additional details, please refer to the source code and configuration files. Feel free to modify the parameters to explore different continual learning setups.
