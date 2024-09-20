from data_loader import load_train_test_data
from feature_extraction import extract_features
from model import build_model
from utils import scale_features, encode_labels
from evaluate import evaluate_model

import numpy as np
from keras.utils import to_categorical
import pandas as pd

# Load data
base_dir = '/home/sujalnath/Downloads/ravdess'
print("Loading and processing data...")
train_data, train_labels, test_data, test_labels = load_train_test_data(base_dir=base_dir)

# Extract features
print("Extracting features...")
train_features = [extract_features(path) for path in train_data]
test_features = [extract_features(path) for path in test_data]

# Column names of the Dataframe
X_train_column_names = X_test_column_names = [f"Feature {i+1}" for i in range(len(train_features[0]))]
y_train_column_names = y_test_column_names = ['emotion', 'intensity', 'gender']

# Preparing the training and testing dataframe (X_train, X_test, y_train, y_test)
X_train_df = pd.DataFrame(train_features, columns=X_train_column_names)
X_test_df = pd.DataFrame(test_features, columns=X_test_column_names)
y_train_df = pd.DataFrame(train_labels, columns=y_train_column_names)
y_test_df = pd.DataFrame(test_labels, columns=y_test_column_names)

# Scaling the input features
X_train_scaled, X_test_scaled = scale_features(X_train_df, X_test_df)

# Extract inputs (features) and outputs (emotion, intensity, gender)
X_train = X_train_df[X_train_column_names].values
y_train = y_train_df['emotion'] + y_train_df['intensity'] + y_train_df['gender']

X_test = X_test_df[X_test_column_names].values
y_test = y_test_df['emotion'] + y_test_df['intensity'] + y_test_df['gender']

# Encode the categorical output variables (emotion, intensity, gender)
y_train_cat, y_test_cat = encode_labels(y_train, y_test)
y_train_cat = to_categorical(y_train_cat, num_classes=30)
y_test_cat = to_categorical(y_test_cat, num_classes=30)

# Reshape X to 3D for LSTM (samples, timesteps, features)
X_train_lstm = np.expand_dims(X_train, axis=1)
X_test_lstm = np.expand_dims(X_test, axis=1)

print("\n\nX_train.shape:",X_train_lstm.shape, end="\n\n")
print("\n\ny_train.shape:",np.shape(y_train_cat), end="\n\n")

input_shape=(1, X_train_lstm.shape[2])
num_classes = np.shape(y_train_cat)[1]

# Building the model
print("Building the model...")
model = build_model(input_shape, num_classes)

# Train the model with multiple outputs
print("Training the model...")
history = model.fit(X_train_lstm,
                    y_train_cat,
                    epochs=10000, batch_size=500, validation_split=0.2, shuffle=True)

# Evaluate the model
evaluate_model(model, X_test_lstm, y_test_cat)