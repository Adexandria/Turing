# Introduction

This project focuses on **CodeBERTa**, a state-of-the-art transformer model for processing developer-written text.  
For context, a **RandomForest-TFIDF** model is also included as a classical baseline.

!!! info "Primary Model"  
    **CodeBERTa is the main model used for predictions and production tasks.**  
    It provides superior accuracy and handles complex text patterns that classical models cannot.

----------

## RandomForest-TFIDF (Baseline)

The **RandomForest-TFIDF** model combines:

1.  **TF-IDF (Term Frequency–Inverse Document Frequency)** for converting text into numerical features.
    
2.  **Random Forest Classifier**, an ensemble of decision trees trained on these features.
    

### Key Points

-   Provides a **simple and interpretable baseline**
    
-   Works well for keyword-driven text
    
-   Fast to train with minimal resources
    

### Role

This model is used mainly for **benchmarking and comparison** to evaluate the improvements offered by CodeBERTa.

----------

## **CodeBERTa (Primary Model)**

!!! tip "CodeBERTa Highlights"  
- **Primary model used in this project** for predictions and evaluation  
- Captures **semantic context and relationships** beyond simple keywords  
- Provides **higher accuracy and F1 scores** compared to classical models  
- Ideal for **production use** in scenarios requiring deep understanding of developer text

### Why It’s Used

CodeBERTa is the **core model** because it effectively handles complex patterns in text and delivers **state-of-the-art performance**.

----------

## Summary

-   **RandomForest-TFIDF:** Classical baseline for comparison
    
-   **CodeBERTa:** **Main model**, advanced transformer providing superior performance
    

The following sections contain the **CodeBERTa model card**, detailing training configuration, evaluation metrics, and usage guidelines.