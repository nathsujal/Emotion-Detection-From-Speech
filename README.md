# Emotion Detection from Speech using LSTM and CNN

This project implements a deep learning model to classify emotions from speech using a combination of Long Short-Term Memory (LSTM) and Convolutional Neural Networks (CNN). The model is trained on the **RAVDESS dataset** and can recognize various emotional states such as happy, angry, sad, and more from audio data.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)


## Project Overview

Speech emotion recognition (SER) is an important task in human-computer interaction. This project leverages deep learning techniques like LSTM and CNN to detect emotions from audio data. By combining these models, the system can effectively capture temporal patterns in speech (via LSTM) and local feature extraction (via CNN).

## Dataset

The **RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)** dataset is used for this project. It consists of 24 actors, each pronouncing two lexically-matched statements across eight emotions with varying intensity.

- **Classes**: Calm, Happy, Sad, Angry, Fearful, Disgust, Surprised, Neutral
- **Audio Format**: .wav files

To download the dataset, visit the [official RAVDESS page](https://zenodo.org/record/1188976#.W1X2F9IzY2w).

## Model Architecture

The deep learning model is designed with two main components:

- **LSTM Layers**: These capture temporal patterns in the speech signal, processing sequential audio frames to maintain the order and context.
- **CNN Layers**: These extract features from the spectrogram, identifying local patterns in the audio signals.

The network structure includes:
- **Two LSTM Layers**: Used to process the sequential data (e.g., mel-spectrogram or MFCC features).
- **Two CNN Layers**: To capture local features.
- **Fully Connected Dense Layers**: For classification into emotion categories.
- **Activation Functions**: ReLU for hidden layers and Softmax for the output layer to classify emotions.
