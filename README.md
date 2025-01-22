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
