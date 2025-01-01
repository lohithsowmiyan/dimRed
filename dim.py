from ezr import *




import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from sklearn.metrics import mean_squared_error, r2_score

def create_autoencoder(input_dim, encoding_dim):
    """
    Create an autoencoder model for dimensionality reduction
    
    Parameters:
    input_dim (int): Number of input features
    encoding_dim (int): Desired number of dimensions after reduction
    """
    # Input layer
    input_layer = Input(shape=(input_dim,))
    
    # Encoder
    encoded = Dense(encoding_dim * 2, activation='relu')(input_layer)
    encoded = Dense(encoding_dim, activation='relu')(encoded)
    
    # Decoder
    decoded = Dense(encoding_dim * 2, activation='relu')(encoded)
    decoded = Dense(input_dim, activation='linear')(decoded)
    
    # Create autoencoder model
    autoencoder = Model(input_layer, decoded)
    encoder = Model(input_layer, encoded)
    
    # Compile model
    autoencoder.compile(optimizer='adam', loss='mse')
    
    return autoencoder, encoder

def evaluate_autoencoder(encoding_dim, epochs=100, batch_size=32):
    """
    Train and evaluate autoencoder for dimensionality reduction
    
    Parameters:
    X (numpy.ndarray): Input features
    y (numpy.ndarray): Target variables
    encoding_dim (int): Desired number of dimensions after reduction
    """
    # Standardize the input data
    d        = DATA().adds(csv(the.train))
    X = np.array(d.rows)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(d.rows)
    
    # Split the data
    X_train, X_test = train_test_split(X_scaled, test_size=0.2, random_state=42)
    
    # Create and train autoencoder
    autoencoder, encoder = create_autoencoder(X.shape[1], encoding_dim)
    autoencoder.fit(X_train, X_train, 
                   epochs=epochs, 
                   batch_size=batch_size, 
                   validation_data=(X_test, X_test),
                   verbose=1)
    
    # Get encoded (reduced) representations
    X_encoded = encoder.predict(X_scaled)
    
    # Get reconstructed data
    X_reconstructed = autoencoder.predict(X_scaled)
    X_reconstructed = scaler.inverse_transform(X_reconstructed)
    
    # Calculate evaluation metrics
    reconstruction_mse = mean_squared_error(X, X_reconstructed)
    reconstruction_r2 = r2_score(X, X_reconstructed)
    
    # Calculate explained variance ratio for encoded features
    original_var = np.var(X, axis=0).sum()
    encoded_var = np.var(X_encoded, axis=0).sum()
    explained_variance_ratio = encoded_var / original_var
    
    return {
        'encoded_features': X_encoded,
        'reconstruction_mse': reconstruction_mse,
        'reconstruction_r2': reconstruction_r2,
        'explained_variance_ratio': explained_variance_ratio,
        'encoder': encoder,
        'autoencoder': autoencoder
    }

print(evaluate_autoencoder(3))