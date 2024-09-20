from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

def build_model(input_shape, num_classes):
    # Build LSTM model
    model = Sequential()
    model.add(LSTM(128, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(64))
    model.add(Dropout(0.2))
    model.add(Dense(32))

    # Output layer
    model.add(Dense(30, activation='softmax'))

    # Compile the model
    model.compile(optimizer='RMSProp',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    print(model.summary())

    return model

build_model((1, 59), 30)