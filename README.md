# optimal-learn-order
Codes for the continual learning project with optimal order model

#### model_train.py
module for model training, including continual learning codes

## Explanations for modules
#### data_analyze.py
module for data extraction and visualization with given .csv file

#### data_process.py
module for pre-process and uploading of datasets from tensorflow dataset

#### group_split.py
module for choosing specific classes(num_task*num_classes) from dataset and split these classes into groups 

#### network.py
module for neuro network generated using flax.nn, including convolutional neuro network and nonlinear neuro network

#### order.py
module for obtaining optimal order based on hamiltonian-path/periphery-core model

#### similarity.py
module for inter-task similarity calculation based on zero-shot/gHg model

## Experiment repository
Some experiment examples codes, details in README.md inside
