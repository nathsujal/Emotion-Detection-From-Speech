import os
import numpy as np

def extract_emotion_label(file_name):
    """
    Extract emotion and emotional intensity from the file name based on RAVDESS naming convention.
    :param file_name: Name of the audio file (e.g., '03-01-03-01-01-01-01.wav')
    :return: label (emotion, emotional_intensity, gender)
    """

    # Split the file name by hyphen
    parts = file_name.split('.')[0].split('-')

    # Emotion and emotional intensity are at specific positions
    emotion_code = parts[2]
    intensity_code = parts[3]
    actor = int(parts[6])

    # Mapping for emotions based on the RAVDESS convention
    emotion_dict = {
        '01': 'neutral',
        '02': 'calm',
        '03': 'happy',
        '04': 'sad',
        '05': 'angry',
        '06': 'fearful',
        '07': 'disgust',
        '08': 'surprised'
    }

    # Mapping for emotional intensity
    intensity_dict = {
        '01': 'normal',
        '02': 'strong'
    }

    # Extract emotion, intensity and gender based on codes
    emotion = emotion_dict.get(emotion_code, 'unkown')
    intensity = intensity_dict.get(intensity_code, 'unkown')
    gender = 'male' if actor%2!=0 else 'female'
    label = np.array([emotion, intensity, gender])

    return label

def get_data(data_dir):
    """
    Load data from the directory structure and return audio paths, labels and emotional intensity.
    :param data_dir: The root directory of the dataset
    :return: data (list of file paths), labels [list of [emotion, intensity, gender]]
    """
    data = []
    labels = []

    for actor_dir in os.listdir(data_dir):
        actor_path = os.path.join(data_dir, actor_dir)
        if os.path.isdir(actor_path):
            for file in os.listdir(actor_path):
                if file.endswith(".wav"):
                    file_path = os.path.join(actor_path, file)
                    label = extract_emotion_label(file)
                    data.append(file_path)
                    labels.append(label)

    return data, np.array(labels)

def load_train_test_data(base_dir):
    """
    Load train and test datasets.
    :param base_dir: The base directory of the RAVDESS dataset (containing 'train' and 'test' folders)
    :return: Train and test data, labels
    """
    train_dir = os.path.join(base_dir, 'train')
    test_dir = os.path.join(base_dir, 'test')

    # Load train data
    train_data, train_labels = get_data(train_dir)

    # Load test data
    test_data, test_labels = get_data(test_dir)

    return train_data, train_labels, test_data, test_labels
