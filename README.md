# Intrusion Detection System Attack
A Wasserstein Generative Adversarial Network (WGAN) is used to fool Intrusion detection systems (IDS) into believing that malicious network traffic is normal traffic.

## Requirements

- Python (>= 3.6)
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
