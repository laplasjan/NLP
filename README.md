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



