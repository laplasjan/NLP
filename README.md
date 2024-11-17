# NLP
**NLP meant to fighting russian misinfo.**

# Key Functions and Pipeline
Text Preprocessing, Support Vector Machine (SVM) Model

# What's new in repository?
In the notebook, I combined all preprocessing steps into a single function and built a weak model, to say the least. The model is weak due to a lack of data and examples of fake news. Given the insufficient number of fake examples, which is a significant problem in classification methods, I used resampling and reduced the classes within the fake category. In the final model, I abandoned the poorly fitted SVM model and replaced it with logistic regression. Accuracy improved from 0.26 to 0.36 after applying the aforementioned methods.

# Steps to be taken in the future:
-> considering new dataset
\
-> considering use of embedings and transfer learning model
\
-> Creating app in FAST API, which will be intuitive and enable to fast check if the text have misinfo.
\
-> Creating an extension for Google Chrome and Firefox wchich automatically blocks misinfo

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

