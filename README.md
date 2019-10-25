# Intrusion Detection System Attack
A Wasserstein Generative Adversarial Network (WGAN) is used to fool Intrusion detection systems (IDS) into believing that malicious network traffic is normal traffic.

## Requirements
- Python (>= 3.7, lower 3.x versions may work but have not been tested)
- [Pipenv](https://github.com/pypa/pipenv)
- [Git Large File Storage](https://git-lfs.github.com/)

## Installation

``` sh
git clone https://github.com/ChrisFugl/Intrusing-Detection-System-Attack
cd Intrusing-Detection-System-Attack
pipenv install --dev
```

### Optional: Jupyter support
Run this command:

``` sh
pipenv run ipython kernel install --user --name=IDSA
```

Use this command to start a Jupyer notebook:

``` sh
pipenv run jupyter notebook
```

Select "IDSA" when creating a new notebook.

## API
Overview of the API from each file. See the comments in the files for documentation of each function.

### data.py
This file contains functions to work with the datasets.

* load_train(): Loads the trainingset (unprocessed).
* load_test(): Loads the testset (unprocessed).
* preprocess(dataframe, \*\*kwargs): Performs preprocessing of a dataset.
* remove_content(dataframe): Removes all content features.
* remove_host_based(dataframe): Removes all host based features.
* remove_time_based(dataframe): Removes all time based features.
* get_content_columns(): Gives the column names of the content features.
* get_host_based_columns(): Gives the column names of the host based features.
* get_time_based_columns(): Gives the column names of the time based features.

Note: All of the functions, except the three getters, return (one or more) Pandas dataframes. Call .to_numpy() on the dataframe object to convert it to a NumPy array.

**Example:**

``` py
training_data = preprocess(load_train())
test_data = preprocess(load_test())
```

Preprocess is not automatically called when loading the datasets. This is intentionally designed that way in order to maintain easy access to the raw datasets.
