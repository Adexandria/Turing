# OWNML MACHINE LEARNING CANVAS

## Value Proposition
This project aims to enhance code comprehension and maintenance by developing a machine learning model capable of automatically classifying code comments.
The primary beneficiaries are software developers, maintainers, and documentation managers. They struggle with inefficient code comprehension and maintenance in large, complex codebases. The sheer volume of unstructured comments makes it difficult and time-consuming to locate specific information, such as design rationale, implementation details, or pending tasks. This slows down development, complicates the onboarding of new team members, and increases the risk of introducing bugs during maintenance.
The comment classifier integrates into the developer's workflow by providing its output as a foundational layer for other software engineering tools. The purpose is to enhance these tools' capabilities, enabling more effective code comprehension and analysis. This allows for functionalities such as targeted searching and filtering of comments by their category, and provides high-level analytics on the codebase's documentation patterns.

## Prediction Task
The task is a multi-label classification task. The entities on which predictions made are code comment sentences belonging to 18 categories across three programming languages (7 for Java, 5 for Python, and 6 for Pharo). The list of labels is defined for each of as follows:

- **Java**: [summary, Ownership, Expand, usage, Pointer, deprecation, rational]
- **Python**: [Usage, Parameters, DevelopmentNotes, Expand, Summary]
- **Pharo**: [Keyimplementationpoints, Example, Responsibilities, Intent, Keymessages, Collaborators]

## Decisions
The ML system’s predictions support a structured process for code comprehension and maintenance. It adds value by filtering, highlighting, or grouping comment sentences by category enhancing code comprehension and maintenance.

Based on the predictions, a developer can decide to selectively filter comments to focus on specific tasks or refactor the code to match the documented design, or update documentation for new features.

## Impact Simulation
The impact of this classification model is measured by its contribution to developer productivity and code comprehension. A correct classification reduces the time required for code navigation and maintenance, reducing the analysis process. The goal is to decrease the number of manual steps a developer must take to understand the function of a specific comment. Conversely, an incorrect classification can lead to minor inefficiencies, forcing the developer to spend additional time verifying the comment's purpose.
The pre-deployment impact simulation consists of evaluating the model's performance (precision, recall, and F1-score) on a test set. The model will be proposed for deployment only if it demonstrates a measurable improvement over current benchmarks.

## Making Predictions
Predictions are performed in Batch mode for scheduled tasks such as periodic model retraining and formal performance evaluation. The system also supports inference whenever an end-user, such as a developer, interacts with the code by viewing a comment or saving a file. These predictions must meet strict low-latency requirements to avoid disrupting the developer’s workflow.

The computation for inference will be performed on standard research hardware, such as a local machine or cloud-based  server, utilizing GPU resources to accelerate the processing phase.

## Data Sources
The sole data source for this project is the official dataset provided for the NLBSE'26 competition, which is hosted on and accessed via the Hugging Face Hub (NLBSE/nlbse26-code-comment-classification). This dataset consists of both entities (code comment sentences for Java, Python, and Pharo) and observed outcomes (the ground-truth, multi-label annotations).

## Data Collection
The NLBSE'26 dataset is made of 1,733 manually labeled class comments and 9,361 sentences from these comments, distributed into various categories specific to each programming language (summary, intent, rationale, etc.). These comments were extracted from 20 open-source projects written in three programming languages: Java, Pharo, or Python. We provide the associated code class (i.e., class name) for each sentence as well as the source code of the software projects. Each comment sentence can belong to one or more categories of the language (from a minimum of 5 to a maximum of 7, depending on the language). Each category represents the type of information that the sentence is conveying.
For the scope of this project, there are no strategies in place for continuous data collection, updates, or active learning. The dataset is considered fixed, and there is no process for sourcing new comments or labels.

## Features
During prediction, the model works with input representations that simplify and enrich the original data. The main text is converted into dense numerical vectors (embeddings) that capture subtle shades of meaning. 
Following this, the cleaned text will be transformed into feature vectors using appropriate natural language processing techniques. The specific methods for preprocessing and feature extraction will be determined and optimized during the experimental phase of the project to identify the most effective representation for the classification task.

## Building Models
The goal is to develop and deploy three distinct, optimized models, one for each programming language (Java, Python, and Pharo).
Rather than following a continuous development cycle, the modeling phase will focus on experimenting with distinct model architectures to identify the most effective design for each language. Each candidate model will be rigorously evaluated, where its average F1-score on a standardized test set will be compared against previous versions. Efficiency metrics such as inference runtime and GFLOPS will also be considered to ensure that performance improvements remain computationally balanced.

## Monitoring
Developer feedback is obtained through APIs and each feedback entry provides a simple validation signal—‘True’ indicates that the model’s prediction aligns with the expected outcome, while ‘False’ highlights a mismatch. System-level metrics are tracked to ensure cost efficiency and meet latency requirements. The impact of the classification system is measured by comparing the success rate of maintenance or refactoring tasks in code using the ML classifications versus code that does not, focusing on code comprehension time and the occurrence of documentation-related bugs.