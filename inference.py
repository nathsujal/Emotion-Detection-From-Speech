from feature_extraction import extract_features

def predict_emotion(model, audio_file):
    features = extract_features(audio_file)
    features = features.reshape(1, -1)
    prediction = model.predict(features)
    return prediction