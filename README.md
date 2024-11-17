# NLP
**NLP meant to fighting russian misinfo.**

# Key Functions and Pipeline
Text Preprocessing, Support Vector Machine (SVM) Model

# Code description - NLP.ipynb notebook

This Python script implements a text classification pipeline for detecting disinformation using a deep learning model. Below is a detailed explanation of its components:

# Libraries and Tools
**scikit-learn**: Used for machine learning operations like TF-IDF vectorization, training a Support Vector Machine (SVM), and evaluating performance.
\
**NLTK**: Provides the Porter Stemmer for stemming text data.
\
**pandas**: Used for loading and manipulating the dataset.
\
**imbalanced-learn (SMOTE)**: For handling class imbalance (though not used in the final implementation).
\

Due to dataset structure, which is very clear I only removes non-alphanumeric characters, converts text to lowercase, stems words using the Porter Stemmer to reduce words to their base forms and use TF-IDF Vectorization.
\
Combines all steps to preprocess the text, vectorize it, train the SVM model, and evaluate its performance on a test set.
# Output
Classification Accuracy: Prints the model's accuracy on the test set.
Label Distribution: Displays the count of examples for each class in the dataset, showing class imbalance.
# Observations
The dataset is highly imbalanced, as evident from the target class distribution.
The classification accuracy of 26% is likely due to the severe class imbalance and limited feature engineering.
\
![Logo](classification_report1)
\
After that, I handle Missing Values: Missing values in Fakes and Disinfo_cases_en are handled using:
\
 - fillna("N/A") for categorical targets (Fakes).
 - Empty strings ("") for text fields (Disinfo_cases_en).
ad removing Rare Classes: classes with fewer than 5 occurrences are reassigned to a new class (-1).

# Class Imbalance Handling
SMOTE (Synthetic Minority Oversampling Technique): Balances the dataset by creating synthetic examples of minority classes during training.
**Thresholding:**
Rare classes are grouped into a catch-all category to simplify classification and improve generalization.

# Model Selection and Training
LinearSVC:
Selected for faster computation compared to SVC.
Trained using resampled data from SMOTE.
Hyperparameters: Regularization parameter C=1.0 ensures a balance between margin maximization and error minimization.
# Performance Evaluation
- Accuracy: The accuracy improves to 36%, reflecting better handling of imbalance compared to the earlier version.
- Class Distribution: Displays class counts for insights into data imbalance and the effectiveness of preprocessing steps.
- 
# Key Features of the Updated Implementation
Robust Data Cleaning: Handles missing values and removes rare classes efficiently.
Imbalance Correction: Applies SMOTE to deal with underrepresented classes.
Efficient Model Training: Uses LinearSVC for quicker training and improved scalability.
Better Vectorization: Introduces feature limits to prevent overfitting.
