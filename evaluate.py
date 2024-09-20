from sklearn.metrics import classification_report
import numpy as np

def evaluate_model(model, X_test, y_test_cat):
    """Evaluate the multi-output model using accuracy and classification report."""
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Convert predictions to the class with the highest probability
    y_pred = np.argmax(y_pred, axis=1)

    # Convert one-hot encoded true labels to class labels
    y_test = np.argmax(y_test_cat, axis=1)
    
    # Print classification reports for each output
    print("Emotion Classification Report:")
    print(classification_report(y_test, y_pred))