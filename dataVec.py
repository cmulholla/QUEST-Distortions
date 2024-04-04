import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer

#vectorizer = SentenceTransformer('intfloat/multilingual-e5-large-instruct', cache_folder="N:\AI\Transformers_cache")
#vectorizer = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
vectorizer = SentenceTransformer("Salesforce/SFR-Embedding-Mistral")
#model = MLPClassifier(random_state=42, alpha=0.25)

data = pd.read_csv('Annotated_data.csv')
distortion_data = data[data['Dominant Distortion'] != 'No Distortion']
distortion_data = distortion_data.reset_index(drop=True)

discreteData = data[['Id_Number', 'Patient Question', 'Dominant Distortion']].copy()
discreteData.loc[discreteData['Dominant Distortion'] != 'No Distortion', 'Dominant Distortion'] = 'Distorted'

X_train, X_test, y_train, y_test = train_test_split(discreteData['Patient Question'], discreteData['Dominant Distortion'], test_size=0.2, random_state=42)

# print the classes in discreteData
print(discreteData['Dominant Distortion'].value_counts())

# print how much space the GPU has in VRAM
print("Available VRAM: ", vectorizer.device)

# vectorize the training data
X_train_vector = vectorizer.encode(X_train.tolist())
X_test_vector = vectorizer.encode(X_test.tolist())

# store the vectors in a dataframe
X_train_vector = pd.DataFrame(X_train_vector)
X_test_vector = pd.DataFrame(X_test_vector)

# store the dataframe in a csv file
X_train_vector.to_csv('data/X_train_vector.csv', index=False)
X_test_vector.to_csv('data/X_test_vector.csv', index=False)

# store the y_train and y_test in a csv file
y_train.to_csv('data/y_train.csv', index=False)
y_test.to_csv('data/y_test.csv', index=False)

