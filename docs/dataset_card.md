# NLBSE'26 Code Comment Classification Dataset

## Dataset Description

### Purpose and Overview
This dataset is provided for the **NLBSE 2026 Tool Competition**. 
The objective is to classify source code comments into specific software engineering categories based on their semantic content.
The problem is formulated as a **Multi-Label Text Classification** task, where a single comment may belong 
to one or more categories simultaneously.

Dataset:https://huggingface.co/datasets/NLBSE/nlbse26-code-comment-classification

The dataset contains **1,733 manually labeled class comments** decomposed into **9,361 sentences** 
extracted from **20 open-source projects**. 
It covers three distinct programming languages: **Java**, **Python**, and **Pharo**.

For each sentence, the dataset includes:

- The natural language text of the comment.
- The associated class name from the source code.
- The source code snippet context (in the combo field).
- Manual annotations validating the information type.

A total of **487 sentences** are strictly multi-label.


### Supported Tasks

- **Multi-Label Classification:** Predicting the correct set of tags for a given comment.
---

## Dataset Structure

### Training and Test Splits

The dataset is divided into six splits, providing a training and testing set for each language:

- Java: `java_train`, `java_test`
- Python: `python_train`, `python_test`
- Pharo: `pharo_train`, `pharo_test`

### Data Instances
An example of a data instance is as follows:

```json
{
  "index": 1,
  "comment_sentence": "this impl delegates to the old filesystem",
  "combo": "this impl delegates to the old filesystem | Abfss.java",
  "partition": "Abfss.java",
  "labels": "[0, 0, 1, 0, 0, 0, 0]",
  "partition": 0
}
```
### Data Fields

- **comment_sentence**: the actual sentence string, which is part of a (multi-line) class comment;
- **combo**: the class name appended to the sentence string, used to train the baselines;
- **partition**: the dataset split in training and testing; 0 identifies training instances, and 1 identifies testing instances, respectively;.
- **labels**: the ground-truth category, it is a binary list that a single sample belongs to. Each sample belongs one or more categories;
- **class**: the class name referring to the source code file where the sentence comes from;
- **index**: unique identifier for the sample within the dataset.

---

### Label Taxonomy
Each programming language has a specific set of categories. 
The labels vector in the data instances corresponds to the indices listed below.

#### Java (7 Classes)
The Java dataset contains 6,595 sentences distributed across the following categories:

- **Index 0 - Summary:** High-level description of the method or class.
- **Index 1 - Ownership:** License, author, or copyright information.
- **Index 2 - Expand:** Detailed explanation or elaboration.
- **Index 3 - Usage:** Examples or instructions on how to use the code.
- **Index 4 - Pointer:** References to other parts of the code or external docs 
- **Index 5 - Deprecation:** Information about deprecated methods.
- **Index 6 - Rational:** Design rationale or reasons for implementation choices.

#### Python (5 Classes)
The Python dataset contains 1,658 sentences distributed across the following categories:

- **Index 0 - Usage:** How to use the function or class.
- **Index 1 - Parameters:** Description of arguments and parameters.
- **Index 2 - DevelopmentNotes:** Notes for developers, TODOs, or fixmes.
- **Index 3 - Expand:** Detailed descriptions.
- **Index 4 - Summary:** Brief overview.

#### Pharo (6 Classes)
The Pharo dataset contains 1,108 sentences distributed across the following categories:

- **Index 0 - Keyimplementationpoints:** Critical implementation details.
- **Index 1 - Example:** Code examples.
- **Index 2 - Responsibilities:** What the class or method is responsible for.
- **Index 3 - Intent:** The purpose of the code.
- **Index 4 - Keymessages:** Important messages sent or received.
- **Index 5 - Collaborators:** Other classes this code interacts with.

---

## Data Statistics and Quality
The quality analysis of the dataset was conducted on the raw data using **Deepchecks**, a framework for validating datasets and 
detecting anomalies in machine learning pipelines. This analysis helped identify potential issues such as class imbalance, 
duplicate entries, conflicting labels, and distributional inconsistencies before preprocessing. 
It also provided insights on the variability in text lengths, comment structures, and multi-label occurrences.

### Class Distribution
The dataset is highly imbalanced across all languages:

- **Java:** Summary and Usage are the majority classes. Deprecation and Rational are rare minority classes.
- **Python:** Usage dominates the distribution.
- **Pharo:** Example is the most frequent class.


### Known Issues

- **Duplicate Entries:** The training set contains redundant comments.
- **Conflicting Labels:** A small percentage of identical comments may map to different labels. 

For example:

"//$NON-NLS-1$,[0 0 0 0 1 0 0]"

"//$NON-NLS-1$,[0 0 0 0 0 1 0]"

- **Code/Text Mix:** Some entries in the text field may consist primarily of commented-out code rather than natural language description.
- **Length Outliers:** The dataset contains text lengths ranging from 1 token to over 800 chars.

---

## Citation
```bibtex
@inproceedings{nlbse26,
  title={NLBSE'26 Tool Competition: Code Comment Classification},
  author={NLBSE Organizers},
  booktitle={Proceedings of the 2026 International Workshop on Natural Language-based Software Engineering},
  year={2026}
}
```

## Dataset Card Authors

Turing Group (SE4AI Course)

## Dataset Card Contact

GitHub: https://github.com/se4ai2526-uniba/Turing