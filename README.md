# Turing

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Developing advanced machine learning approaches for multi-label code comment classification using the provided dataset of 9,361 code comment sentences.

## Project Organization
```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         turing and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── turing   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes turing a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

# Background

Code comments play a vital role in software development. They help explain complex algorithms, clarify design decisions, and improve overall code maintainability. Automatically classifying these comments can significantly enhance **code comprehension, documentation quality, and developer productivity**.

The **NLBSE'26 competition** provides datasets for **Java, Python, and Pharo**, along with baseline models built using **Sentence Transformers**. This project leverages these datasets to build a robust **code comment classification system** using modern language models.

--------

This guide focuses on setting up the environment and executing the model training pipeline using the dataset.

## 1. Environment Setup

Start by making sure you’ve got Python 3.12 installed. This project uses uv for dependency management, which keeps things fast and smooth.

- To set up the virtual environment, just run:
```bash
make create_environment
````

* To activate it:

  * On Windows:

  ```bash
  .\.venv\Scripts\activate
  ```

  * On Unix/macOS:

  ```bash
  source ./.venv/bin/activate
  ```

* Next, install all the dependencies:

```bash
make requirements
```

That command takes care of everything listed in pyproject.toml, including MLflow, Torch, and Transformers.

## 2. Running the Training Pipeline

The main training script covers the full process—model setup, training for each language (Java, Python, Pharo), and evaluation.

* Launch the training script like this:

```bash
python turing/modeling/train.py --model codeberta
```

You can choose from several models: codeberta (default), graphcodebert, tinybert, or randomforest.



