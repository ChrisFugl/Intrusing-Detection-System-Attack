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

### Jupyter Support (optional)
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

## Tests
All tests are located in the *tests* directory. Use the following command to run all tests:

``` sh
python -m unittest discover -s tests
```

## Structure
The following is proposed for how to structure the codebase. The specific API of each file is to be decided after agreeing on a structure.

**data/**: Directory containing the dataset.

* **data/KDDTest.csv**: NSL-KDD full testset.
* **data/KDDTrain.csv**: NSL-KDD full trainingset.

**data.py** (already implemented): All functions needed to load and preprocess the data.

**ids/**: Module containing model implementations for each of the IDS algorithms. Each model implementation should expose a class with (at least) a train and a test method. These methods are to be used by train_ids.py, test_ids.py, and test_wgan.py.

* **decision_tree.py**: Implementation of decision tree based IDS.
* **k_nearest_neighbours.py**: Implementation of K-nearest neighbours based IDS.
* **linear_regression.py**: Implementation of linear regression based IDS.
* **multi_layer_perceptron.py**: Implementation of multi-layer perceptron (fully connected neural network) based IDS.
* **naive_bayes.py**: Implementation of Naive Bayes based IDS.
* **random_forest.py**: Implementation of random forest based IDS.
* **support_vector_machine.py**: Implementation of support vector machine based IDS.

**model.py**: WGAN should be implemented as two classes (Generator and Discriminator) in this file.

**test_ids.py**: Script for evaluating the performance of a pretrained IDS model. The IDS model parameters should be accepted as a command line argument. For example:

``` sh
python test_ids.py --ids saved_models/{IDS}.pt
```

**test_wgan.py**: Script for testing the performance of a pretrained WGAN against one of the pretrained IDS models. The script should accept command line arguments to specify which WGAN- and IDS parameters to use. For example:

``` sh
python test_wgan.py --wgan saved_models/{WGAN}.pt --ids saved_models/{IDS}.pt
```

**train_ids.py**: Script for training the IDS models on the NSL-KDS trainingset. Command line arguments should be used to control which IDS model to train and to control hyperparameters. For example:

``` sh
python train_ids.py --ids mlp --learning_rate 0.0001
```

**train_wgan.py**: Script for training the WGAN on the NSL-KDD trainingset. Command line arguments should be used to control the hyperparameters. For example:

``` sh
python train_wgan.py --learning_rate 0.0001
```

See [this site](https://pytorch.org/tutorials/beginner/saving_loading_models.html) for how to save and load models in PyTorch.

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
training_data, training_class, training_attack_class = preprocess(load_train())
test_data, test_class, test_attack_class = preprocess(load_test())
```

Preprocess is not automatically called when loading the datasets. This is intentionally designed that way in order to maintain easy access to the raw datasets.
