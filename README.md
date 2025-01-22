Codes for the continual learning project with optimal order model

Thank you for using this code! If you find it useful in your research or work, please cite the associated article:

## Citation
Please refer to the following article if you use this code:

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

## Explanations for modules

#### continual_model.py
module for model training, including continual learning codes

#### dataset_process.py
module for pre-process and uploading tensorflow dataset (ex. cifar10, cifar100, fashion_mnist)

#### file_extract.py
module for data extraction with given .csv files which save labels and accuracies

#### group_split.py
module for choosing specific classes(num_task*num_classes) from specific dataset and split these classes into num_task groups 

#### network.py
module for neuro network generated using flax.nn, including convolutional neuro network and nonlinear neuro network

#### opt_order.py
module for obtaining optimal order based on hamiltonian-path/periphery-core model

#### similarity.py
module for inter-task similarity calculation based on zero-shot/-gHg model

## Examples repository
Some experiment examples codes, details in README.md inside

## General Parameters

The following parameters are used in this machine learning model:

| Parameter Name     | Description                                   | Default Value |
|--------------------|-----------------------------------------------|---------------|
| `learning_rate`    | Learning rate for the optimizer.             | `0.001`       |
| `batch_size`       | Number of samples per batch during training. | `4`        |
| `ghg_batch_size`       | Number of samples per batch during similarity calculation with -ghg model. | `256`        |
| `num_epochs`       | Number of training epochs.                   | `5`         |
| `optimizer`        | Optimizer used for training.                 | `Adam`        |
| `loss_function`    | Loss function used to train the model.       | `CrossEntropy`|
| `shuffle_size`     | Size of the buffer used for shuffling the dataset.| `1000`       |
| `num_perm`       | Number of permutations to represent average performance in continual learning.            | `6`   |

