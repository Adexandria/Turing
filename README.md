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

## Background

Code comments play a vital role in software development. They help explain complex algorithms, clarify design decisions, and improve overall code maintainability. Automatically classifying these comments can significantly enhance **code comprehension, documentation quality, and developer productivity**.

The **NLBSE'26 competition** provides datasets for **Java, Python, and Pharo**, along with baseline models built using **Sentence Transformers**. This project leverages these datasets to build a robust **code comment classification system** using modern language models.

--------

## Production Workflow

### Milestone 1: ML Canvas and AI Risk Analysis

This milestone focuses on aligning technical objectives with business goals and assessing potential risks.

#### ML Canvas

The **ML Canvas** is a strategic tool that bridges the gap between technical work and business goals. It provides a **high-level overview** of the machine learning project, including:

- Target users and stakeholders
- Expected outcomes
- Model predictions
- Data sources and collection methods
- Features used

The canvas facilitates collaboration between technical and non-technical team members, ensuring the ML solution aligns with real-world objectives.

**📄 Documentation**: [ML Canvas](https://se4ai2526-uniba.github.io/Turing/mlcanvas/)

#### AI Risk Analysis

The **AI Risk Analysis** evaluates potential risks associated with deploying machine learning models. Key areas of focus include:

- **Data bias** and fairness issues
- **Security vulnerabilities**
- Impact of **incorrect predictions** on developer productivity and code maintenance

This step helps ensure that the ML system is **safe, reliable, and ethically sound**.

**📄 Documentation**: [AI Risk Analysis](https://se4ai2526-uniba.github.io/Turing/ai-risk-analysis/)

--------

### Milestone 2: DVC and MLflow Integration

**Pull Request**: [#19](https://github.com/se4ai2526-uniba/Turing/pull/19)

**Goal:** Build a reproducible and trackable machine learning workflow.

- **Reproducible Pipeline:** Developed a modeling pipeline using **DVC (Data Version Control)** to manage datasets, intermediate outputs, and model artifacts.
- **Experiment Tracking:** Integrated **MLflow** to log model parameters, metrics, and artifacts, enabling **easy comparison and reproducibility** of experiments.

#### Merged Pull Requests:
- **DVC Integration**: [#14](https://github.com/se4ai2526-uniba/Turing/pull/14)
- **MLflow Integration**: [#18](https://github.com/se4ai2526-uniba/Turing/pull/18)

--------

### Milestone 3: Data and Code Quality

**Pull Request**: [#40](https://github.com/se4ai2526-uniba/Turing/pull/40)

This milestone focused on **data integrity, code quality, and robust testing**.

#### Code Structure & CLI
- Refactored main scripts (e.g., `dataset.py`) into **modular, class-based CLI applications**.
- Structured the training pipeline in a clear, modular way (`modeling/train.py`) for maintainability and reusability.

#### Data Versioning & Validation
- Implemented **Deepchecks** to monitor data quality and validate datasets for consistency.

#### Code Quality (CI)
- Used **Ruff** for linting and code formatting.
- Enforced coding standards through **GitHub Actions** to ensure automated quality checks.

#### Testing (CI)
- Set up **Pytest** for unit testing.
- Added behavioral tests to validate model logic, robustness, and expected performance.

--------

### Milestone 4: API Implementation

**Pull Request**: [#64](https://github.com/se4ai2526-uniba/Turing/pull/64)

In this milestone, we developed a **HTTP POST endpoint** to send text to the deployed **MLflow model** for predictions.

Additional features implemented:

- **CodeBERTa Model:** Integrated a transformer-based model to improve **accuracy and F1 score** over classical baselines.
- **Test Report Generator:** Automated generation of **data and code quality reports**.
- **Model Unit Tests:** Comprehensive unit tests for all models to ensure correctness.
- **API & Model Documentation:** Provided documentation on API usage, endpoint behavior, and model information.
      **📄 Documentation**: [Model API User Guide](https://se4ai2526-uniba.github.io/Turing/model-api/)
- **Model Card:** Added a detailed model card describing the architecture, training data, intended uses, evaluation metrics, and environmental impact. **📄 Documentation**: [Model Card](https://se4ai2526-uniba.github.io/Turing/model_card/)
- **Dataset Card:** Provides information about the dataset’s contents, intended context, creation process, and other relevant considerations for users. **📄 Documentation**: [Dataset Card](https://se4ai2526-uniba.github.io/Turing/dataset_card/)

--------

### Milestone 5: Containerization, CI/CD, HF Spaces and GUI

**Pull Request:** [#77](https://github.com/se4ai2526-uniba/Turing/pull/77)

This milestone focuses on **containerized deployment, automation, and providing an interactive GUI** for model inference.

#### MLflow Model Tagging
- Implemented a **model tagging mechanism in MLflow**, assigning tags to each model based on attributes such as language, dataset, and model type.  
- The tag `best_model` is automatically assigned to the best-performing model for each programming language immediately after training any new model. The tag is removed from the previous best model, ensuring that only the current best model has the `best_model` tag.

**📄 Documentation**: [Tags](https://se4ai2526-uniba.github.io/Turing/pr-preview/pr-77/model_selection/)

#### Containerization and Deployment
- Added a `Dockerfile` and `docker-compose.yml` to enable containerized deployment of the FastAPI application, including user permissions, dependency installation, and environment variable configuration.
- The Docker image is **autonomous in selecting models**: thanks to the `best_model` MLflow tag, when the container starts, it automatically downloads the best model for each language and uses it for serving future API requests. This ensures that the system always serves the most performant models.
- Updated `.dockerignore` to exclude unnecessary files and directories from Docker builds, improving build performance and security.

#### Automation, CI/CD and HF Spaces
- Built and uploaded the Docker image to a Hugging Face Space to enable containerized deployment.  
  **API:** [https://turing-team-turing-space.hf.space/docs](https://turing-team-turing-space.hf.space/docs)
- Introduced a GitHub Actions workflow (`.github/workflows/push-folders.yml`) to automatically sync the `turing` folder to the Hugging Face Space when changes are pushed to relevant branches.

#### Model Inference GUI
- Implemented a graphical user interface for model classification inference using the Gradio library.
- The GUI is accessible via a localhost endpoint and is included in the Docker image for seamless deployment.  
  **GUI:** [https://turing-team-turing-space.hf.space/gradio](https://turing-team-turing-space.hf.space/gradio)
- Added functionality for **user feedback**: after performing inference, users can select the correct category from a dropdown menu. Feedback is automatically saved to a CSV file for future analysis and model improvement.
- Added detailed documentation for the GUI (`docs/gui.md`), including user instructions, technical architecture, and feedback data storage.

**📄 Documentation**: [GUI](https://se4ai2526-uniba.github.io/Turing/pr-preview/pr-77/gui/)

--------