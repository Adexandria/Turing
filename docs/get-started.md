# Get Started

This section explains how to download the dataset and run the full model training and evaluation workflow. You can reproduce the pipeline either using **DVC** or by running each step manually.

----------

## Download the Dataset

To begin training the model, you must first download the required dataset.

1.  Navigate to the project’s `data` directory.
    
2.  Use **DVC** to pull the raw dataset:
    

```dvc pull```

!!! note  
    Ensure DVC is installed and properly configured with access to the remote storage.

----------

## Train and Evaluate the Model

Model training can be executed in two ways:

1.  **Using the DVC pipeline** (recommended)
    
2.  **Running each step manually**
    

Both approaches will produce the extracted features, train the model, and generate evaluation metrics.

----------

## Option 1: Run the DVC Pipeline (Recommended)

To reproduce the entire workflow automatically, simply run:

`dvc repro` 

This command executes every stage defined in the DVC pipeline, from preprocessing to model training and evaluation.

!!! tip  
    Use `dvc dag` to visualize the dependency graph of the pipeline.

----------

## Option 2: Run the Steps Manually

If you prefer to execute each stage independently, follow the steps below.

### 1. Convert Dataset to Parquet/CSV

Converts the dataset into the required format:

```python turing/CLI_runner/run_dataset.py parquet-to-csv``` 

### 2. Extract Features

Runs the feature extraction process:

```python -m turing.features --use-combo-feature```

!!! note  
    The `--use-combo-feature` flag enables combined feature extraction for improved model performance.

### 3. Train and Evaluate the Model

Finally, train the model and generate evaluation results:

```python turing/modeling/train.py```
