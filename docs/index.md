
# Turing – Code Comment Classification with LLMs

## Background

Code comments play a vital role in software development. They help explain complex algorithms, clarify design decisions, and improve overall code maintainability. Automatically classifying these comments can significantly enhance **code comprehension, documentation quality, and developer productivity**.

The **NLBSE'26 competition** provides datasets for **Java, Python, and Pharo**, along with baseline models built using **Sentence Transformers**. This project leverages these datasets to build a robust **code comment classification system** using modern language models.

----------

## Production Workflow

### Milestone 1: ML Canvas and AI Risk Analysis

#### ML Canvas

The **ML Canvas** is a strategic tool that bridges the gap between technical work and business goals. It provides a **high-level overview** of the machine learning project, including:

-   Target users and stakeholders
    
-   Expected outcomes
    
-   Model predictions
    
-   Data sources and collection methods
    
-   Features used
    

The canvas facilitates collaboration between technical and non-technical team members, ensuring the ML solution aligns with real-world objectives.

#### AI Risk Analysis

The **AI Risk Analysis** evaluates potential risks associated with deploying machine learning models. Key areas of focus include:

-   **Data bias** and fairness issues
    
-   **Security vulnerabilities**
    
-   Impact of **incorrect predictions** on developer productivity and code maintenance
    

This step helps ensure that the ML system is **safe, reliable, and ethically sound**.

----------

### Milestone 2: DVC and MLflow Integration

**Goal:** Build a reproducible and trackable machine learning workflow.

-   **Reproducible Pipeline:** Developed a modeling pipeline using **DVC (Data Version Control)** to manage datasets, intermediate outputs, and model artifacts.
    
-   **Experiment Tracking:** Integrated **MLflow** to log model parameters, metrics, and artifacts, enabling **easy comparison and reproducibility** of experiments.
    

----------

### Milestone 3: Data and Code Quality

This milestone focused on **data integrity, code quality, and robust testing**.

#### Code Structure & CLI

-   Refactored main scripts (e.g., `dataset.py`) into **modular, class-based CLI applications**.
    
-   Structured the training pipeline in a clear, modular way (`modeling/train.py`) for maintainability and reusability.
    

#### Data Versioning & Validation

-   Implemented **Deepchecks** to monitor data quality and validate datasets for consistency.
    

#### Code Quality (CI)

-   Used **Ruff** for linting and code formatting.
    
-   Enforced coding standards through **GitHub Actions** to ensure automated quality checks.
    

#### Testing (CI)

-   Set up **Pytest** for unit testing.
    
-   Added behavioral tests to validate model logic, robustness, and expected performance.
    

----------

### Milestone 4: API Implementation

In this milestone, we developed a **HTTP POST endpoint** to send text to the deployed **MLflow model** for predictions.

Additional features implemented:

-   **CodeBERTa Model:** Integrated a transformer-based model to improve **accuracy and F1 score** over classical baselines.
    
-   **Test Report Generator:** Automated generation of **data and code quality reports**.
    
-   **Model Unit Tests:** Comprehensive unit tests for all models to ensure correctness.
    
-   **API & Model Documentation:** Provided detailed documentation on API usage, endpoint behavior, and model information.


