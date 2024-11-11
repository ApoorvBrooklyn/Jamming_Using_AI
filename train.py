import tensorflow as tf
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Dropout, Flatten, Input
import numpy as np
import h5py
import os
import json

class JammingDetector:
    def __init__(self, input_shape=(1000, 1)):
        self.input_shape = input_shape
        self.model = self._create_model()
        
    def _create_model(self):
        """Create the neural network model with explicit input shape"""
        model = Sequential([
            Input(shape=self.input_shape),  # Explicit input layer
            Conv1D(32, 3, activation='relu'),
            MaxPooling1D(2),
            Conv1D(64, 3, activation='relu'),
            MaxPooling1D(2),
            Conv1D(64, 3, activation='relu'),
            Flatten(),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    def train(self, train_data_path, val_data_path, epochs=50, batch_size=32):
        """Train the model on generated dataset"""
        # Load training data
        with h5py.File(train_data_path, 'r') as f:
            X_train = f['jammed_signals'][:]
            y_train = f['labels'][:]
            
        # Load validation data
        with h5py.File(val_data_path, 'r') as f:
            X_val = f['jammed_signals'][:]
            y_val = f['labels'][:]
        
        # Reshape data for CNN
        X_train = X_train.reshape(-1, self.input_shape[0], 1)
        X_val = X_val.reshape(-1, self.input_shape[0], 1)
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size
        )
        
        self.history = history.history
        return history.history
    
    def save_model(self, model_dir):
        """Save the trained model and training history"""
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model architecture and weights separately
        model_path = os.path.join(model_dir, 'model.h5')
        self.model.save(model_path, save_format='h5')
        
        # Save model configuration
        config_path = os.path.join(model_dir, 'model_config.json')
        config = {
            'input_shape': self.input_shape,
            'model_config': self.model.get_config()
        }
        with open(config_path, 'w') as f:
            json.dump(config, f)
        
        # Save training history
        if hasattr(self, 'history'):
            history_path = os.path.join(model_dir, 'training_history.json')
            with open(history_path, 'w') as f:
                json.dump(self.history, f)

if __name__ == "__main__":
    # Create and train model
    detector = JammingDetector()
    history = detector.train("data/train_data.h5", "data/val_data.h5")
    detector.save_model("models")