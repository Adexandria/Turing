# Model Card for CodeBERTa-Comment-Classification: Java

<!-- Provide a quick summary of what the model is/does. -->

A fine-tuned CodeBERTa model designed for the multi-label classification of source code comments, capable of identifying various comment types (e.g., Summary, Ownership, Deprecation, ...) in Java programming language.

## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->

This model is a fine-tuned version of huggingface/CodeBERTa-small-v1 tailored for classifying code comments. It utilizes a Transformer-based architecture (AutoModelForSequenceClassification) to perform multi-label classification, meaning a single comment can be assigned multiple categories simultaneously.

The model classifies comments into 7 distinct categories.

**Model info:**

- **Developed by:** Turing group
- **Model type:** Multi-label classification
- **Language(s) (NLP):** English (code comments)
- **Finetuned from model:** huggingface/CodeBERTa-small-v1

### Model Sources

<!-- Provide the basic links for the model. -->

- **Repository:** [Turing group repository](https://github.com/se4ai2526-uniba/Turing)
- **Demo:** [API Link](http://127.0.0.1:8000/docs#/default/predict_predict_post)

## Uses

<!-- Address questions around how the model is intended to be used, including the foreseeable users of the model and those affected by the model. -->

### Direct Use

<!-- This section is for the model use without fine-tuning or plugging into a larger ecosystem/app. -->

The model is intended for the automatic analysis of source code repositories. It can be used to categorize comments to help developers understand codebases. The model takes a string (comment) as input and outputs a binary vector representing the active categories.

### Downstream Use

<!-- This section is for the model use when fine-tuned for a task, or when plugged into a larger ecosystem/app -->

This model is designed to be integrated into larger software engineering ecosystems. Below are specific downstream tasks enabled by the model's multi-label classification capabilities:

- **IDE Plugin Integration:** Filtering and organizing comments within editors.
- **Maintenance Search Tools:** Filtering codebases by comment category.
- **Automated Documentation Auditing:** Enhancing code review workflows with intelligent comment analysis.
- **Semantic Code Navigation:** Enabling advanced search capabilities beyond simple keyword matching.

### Out-of-Scope Use

<!-- This section addresses misuse, malicious use, and uses that the model will not work well for. -->

This model is not designed for code generation or completion. It should not be used for general-purpose natural language tasks unrelated to software engineering contexts (e.g., sentiment analysis of social media).

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

The model is trained on the **NLBSE'26 Code Comment Classification** dataset, which consists of comments extracted from open-source repositories. Consequently, the model may inherit biases present in the source code of these projects, such as:

- **Domain Bias:** A bias towards specific programming domains or styles prevalent in the collected repositories.
- **Label Noise:** Potential inconsistencies in the ground truth labels inherent to crowd-sourced or automatically mined datasets.

### Recommendations

<!-- This section is meant to convey recommendations with respect to the bias, risk, and technical limitations. -->

Users should validate the model's predictions on their specific codebase, as comment conventions vary significantly between projects. It is recommended to manually audit a sample of predictions before integrating the model into automated decision-making pipelines.

## Training Details

### Training Data

<!-- This should link to a Dataset Card, perhaps with a short stub of information on what the training data is all about as well as documentation related to data pre-processing or additional filtering. -->

The model utilizes the [**NLBSE'26 Code Comment Classification** dataset](https://huggingface.co/datasets/NLBSE/nlbse26-code-comment-classification) (hosted on Hugging Face).

For detailed information about the dataset, see the [Dataset Card](install.md).

- **Source:** The dataset consists of source code comments extracted from various open-source repositories.
- **Composition:** It contains labeled data for **multi-label classification**, covering multiple programming languages (Java, Python, Pharo).
- **Data Quality:** The training set is a curated subset of the original data. It has been processed to remove duplicates, ambiguous labels, and non-informative comments (noise) to ensure high-quality supervision.
- **Class Balance:** Strategies were employed to mitigate class imbalance, ensuring that under-represented categories have sufficient examples for training.

### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

#### Preprocessing

The training pipeline implements a **"Smart Cleaning"** and **"Safe Augmentation"** strategy to prepare the data before it reaches the model:

**1. Data Cleaning & Filtering:**
Prior to tokenization, the raw text undergoes heuristic filtering:

- **Deduplication:** Removal of exact duplicates and semantic conflicts (identical text with different labels).
- **Noise Reduction:** Comments are filtered out if they are too short (< 2 tokens), too long, or consist primarily of code symbols rather than natural language.
- **Normalization:** Text is converted to lowercase and comment markers (e.g., `//`, `/*`) are stripped.

**2. Data Augmentation:**
To handle imbalance, a **Safe Augmentation** technique is applied to minority classes:

- **Method:** Synonym replacement via WordNet and random case injection.
- **Safety:** A strict "protected list" of code keywords (e.g., `return`, `if`, `void`) is used to prevent the augmentation process from corrupting the semantic logic of code snippets.

**3. Model Input Processing:**

- **Tokenization:** Uses the `AutoTokenizer` for `CodeBERTa-small-v1`.
- **Encoding:** Labels are transformed into one-hot encoded vectors to support multi-label classification.


#### Training Hyperparameters

- **Model:** `huggingface/CodeBERTa-small-v1`.
- **Optimizer:** AdamW (`adamw_torch`).
- **Learning Rate:** 1e-5.
- **Batch Size:** 16 (Train), 64 (Eval).
- **Epochs:** 15 (with Early Stopping patience of 3).
- **Precision:** Mixed Precision (fp16) enabled for CUDA devices.
- **Max Length:** 128 tokens.

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data, Factors & Metrics

#### Testing Data

<!-- This should link to a Dataset Card if possible. -->

The testing data corresponds to the official test split of the NLBSE'26 dataset. Critically, **no augmentation** was applied to the test set to ensure evaluation against realistic, unmodified data distributions.

#### Factors

<!-- These are the things the evaluation is disaggregating by, e.g., subpopulations or domains. -->

Evaluation is performed separately for each target programming language, as the label definitions and counts differ between languages.

#### Metrics

<!-- These are the evaluation metrics being used, ideally with a description of why. -->

- **F1 Score:** The primary metric used for checkpoint selection.
- **Accuracy:** Subset accuracy.
- **Decision Threshold:** A sigmoid threshold of **0.5** is used for binary predictions.

### Results

| Label | Category Name | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| 0 | Summary | 0.86 | 0.92 | 0.89 |
| 1 | Ownership | 1.00 | 1.00 | 1.00 |
| 2 | Expand | 0.30 | 0.16 | 0.21 |
| 3 | Usage | 0.93 | 0.77 | 0.84 |
| 4 | Pointer | 0.76 | 0.94 | 0.84 |
| 5 | Deprecation | 0.67 | 0.60 | 0.63 |
| 6 | Rational | 0.24 | 0.16 | 0.19 |
|  | **Weighted Avg** | **0.81** | **0.80** | **0.80** |

**Test Metrics:**

- **Accuracy:** 0.7802
- **F1-Score (Micro):** 0.8127
- **F1-Score (Weighted):** 0.7975

#### Efficiency & Computational Cost

- **Average Runtime:** `1.770s`
- **Computational Cost:** `14,781 GFLOPs`

## Environmental Impact

<!-- Total emissions (in grams of CO2eq) and additional considerations, such as electricity usage, go here. Edit the suggested text below accordingly -->

- **Hardware Type:** The code is designed to run on **NVIDIA GPUs** (CUDA) to leverage hardware acceleration.
- **Energy Optimization:** Mixed Precision training (`fp16`) was enabled to reduce memory usage and energy consumption on compatible hardware.

## Citation

<!-- If there is a paper or blog post introducing the model, the APA and Bibtex information for that should go in this section. -->

**BibTeX:**

```bibtex
@inproceedings{nlbse26,
  title={NLBSE'26 Tool Competition: Code Comment Classification},
  author={NLBSE Organizers},
  booktitle={Proceedings of the 2026 International Workshop on Natural Language-based Software Engineering},
  year={2026}
}
```

## Model Card Authors

Turing Group (SE4AI Course)

## Model Card Contact

GitHub: https://github.com/se4ai2526-uniba/Turing