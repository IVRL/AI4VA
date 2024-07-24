# Import necessary libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Flatten, Dense, Input
from tensorflow.keras.models import Model

# a simple baseline model for depth ordering
def create_baseline_model(input_shape):
    inputs = Input(shape=input_shape)
    
    # Convolutional layers for feature extraction
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    
    # Flatten and fully connected layers for prediction
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    inter_depth = Dense(1, name='inter_depth')(x)
    intra_depth = Dense(1, name='intra_depth')(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=[inter_depth, intra_depth])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model

# Example of model training
model = create_baseline_model((128, 128, 3))
# Assume X_train, y_train_inter, and y_train_intra are preprocessed datasets
# y_train = {'inter_depth': y_train_inter, 'intra_depth': y_train_intra}
# model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Save the model
model.save('best_model.h5')
