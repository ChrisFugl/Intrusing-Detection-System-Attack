# Intrusion Detection System Attack
A Wasserstein Generative Adversarial Network (WGAN) is used to fool Intrusion detection systems (IDS) into believing that malicious network traffic is normal traffic.

## Requirements
Please make sure that you have the following requirements installed on your system:

- Python (>= 3.7, lower 3.x versions may work but have not been tested)
- [Git Large File Storage](https://git-lfs.github.com/)

## Installation
First clone the project.

``` sh
git clone https://github.com/ChrisFugl/Intrusing-Detection-System-Attack
cd Intrusing-Detection-System-Attack
```

**Note:** We recommend that you install the Python packages in a virtual environment. See the next section for how to do this, and then proceed with the rest of this section afterwards.

``` sh
pip install -r requirements.txt
```

### Virtual Environment (optional)
A virtual environment helps you to avoid that Python packages in this project does not conflict with other Python packages in your system. Follow the instructions [on this site](https://virtualenv.pypa.io/en/stable/installation/) to install the virtualenv package, which enables you to create virtual environments.

Once virtualenv is installed, you will need to run the following commands to setup a virtual environment for this project.

``` sh
virtualenv env
```

You may want to add the flag "--python python3" in case your default Python interpreter is not at version 3 (run ```python --version``` to check the Python version):

``` sh
virtualenv --python python3 env
```

Either of the previous two commands will create a directory called *env* in the project directory. You need to run the following command to make use of the virtual environment.

``` sh
source env/bin/activate
```

You are now up an running with the virtual environment. Run the following command when you want to exit this environment.

``` sh
deactivate
```

### Jupyter support (optional)
The commands in this section should be run from inside of a virtual environment. Note that you only need to do these steps if you are using a virtual environment.

Run this command:

``` sh
python -m ipykernel install --user --name=IDSA
```

Use this command to start a Jupyter notebook:

``` sh
jupyter notebook
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
