Description of the Code
This project demonstrates the use of BERT (Bidirectional Encoder Representations from Transformers) for extracting text embeddings and performing sequence classification on a dataset. Below is a breakdown of the steps and components involved in the code:

1. Importing Necessary Libraries
The code begins by importing necessary libraries:

pandas: Used for data manipulation and reading CSV files.
transformers: A library for loading pre-trained BERT models and tokenizers.
torch: The PyTorch framework is used for model inference and handling tensors.
numpy: For numerical operations, particularly handling BERT embeddings.
google.colab: Mounts Google Drive to access dataset files stored in Google Drive.
import pandas as pd
from transformers import BertTokenizer, BertModel, BertForSequenceClassification
import torch
import numpy as np
from google.colab import drive
2. Loading the Dataset
The dataset is loaded from Google Drive and consists of two columns:

Disinfo_cases_en: Contains misinformation cases in English.
Fakes: Contains the corresponding fake claims or disinformation statements.
The dataset is read as a CSV file into a pandas DataFrame.

drive.mount('/content/drive')
file_path = '/content/drive/My Drive/data.csv'
data = pd.read_csv(file_path)
3. Preparing Text for BERT
The text from both columns (Disinfo_cases_en and Fakes) is combined into a single string for each row. This combined text is used as input for the BERT model.

col1 = data["Disinfo_cases_en"].astype(str)
col2 = data["Fakes"].astype(str)
text = col1 + " " + col2
4. BERT Tokenization and Embedding Extraction
A BERT tokenizer is loaded to tokenize the text and convert it into input tensors. The BERT model (base uncased version) is then used to generate embeddings. The embeddings are extracted from the [CLS] token's final hidden state, representing a compact representation of the text.


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def get_bert_embeddings(text):
    if isinstance(text, str):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return cls_embedding.squeeze().numpy()
    else:
        return np.nan
The embeddings are then applied to the text data and stored as a new column in the DataFrame.

embeddings = [get_bert_embeddings(t) for t in text]
data['embedding'] = embeddings
5. Sequence Classification with BERT
In addition to extracting embeddings, the code uses another pre-trained model (nlptown/bert-base-multilingual-uncased-sentiment) for sequence classification. This model predicts sentiment scores for the given text.

The text is tokenized, and the model's output logits (raw classification scores) are converted to probabilities using softmax.

model_classification = BertForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
inputs_classification = tokenizer(text.tolist(), return_tensors="pt", truncation=True, padding=True, max_length=128)

with torch.no_grad():
    classification_outputs = model_classification(**inputs_classification)

logits = classification_outputs.logits
probabilities = torch.nn.functional.softmax(logits, dim=-1)
The output is a set of probabilities for each class, reflecting sentiment scores across multiple categories (e.g., negative to positive sentiment).

print("Probabilities for each class:", probabilities.numpy())
6. Visual Representation of the Output
The final output consists of a DataFrame containing:

The original misinformation cases and their fake claims.
The BERT embeddings representing the text as vectors.
The classification probabilities for sentiment analysis.
This dataset can be used for further analysis, clustering, or training other models.

Output Example
Disinfo_cases_en	Fakes	embedding
Kyiv forces continue to bomb the territories...	Kyiv ignores fulfillment of the Minsk Agreements	[-0.8118986, -0.086330086, 0.19561116, -0.4455...]
After the Maidan putsch in April 2014...	Maidan led to the separation of Donetsk and Lugansk	[-0.5667894, -0.28954983, -0.29709616, -0.2573...]
...	...	...
The embedding column contains the numerical vector representation of the text after passing through the BERT model, which can be used for various tasks such as clustering, similarity analysis, or further classification.

Conclusion
This project demonstrates how to use pre-trained BERT models to:

Generate text embeddings for information retrieval or clustering.
Perform sequence classification (sentiment analysis) on disinformation-related texts.
With this approach, the model can be easily extended for other natural language processing tasks, such as topic modeling, summarization, or custom classification tasks.
