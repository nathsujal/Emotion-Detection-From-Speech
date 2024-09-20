from data_loader import load_train_test_data
from feature_extraction import extract_features
import pandas as pd
import numpy as np

train_data, train_labels, test_data, test_labels = load_train_test_data('/home/sujalnath/Downloads/ravdess')

train_features = [extract_features(file_path) for file_path in train_data]

train = np.hstack([train_features, train_labels])

df = pd.DataFrame(train)

column_names = np.hstack([
    [f"Feature {i+1}" for i in range(len(train_features[0]))],
    ['emotion', 'intensity', 'gender']
    ])

df.columns = column_names

print("\ntrain labels\n", train_labels)
print("\ntrain features\n", train_features[0])
print("\nDataFrame\n")
print(df.head)