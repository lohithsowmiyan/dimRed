from ezr import *
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE



def compute_PCA(X, n_components):
        """
        Perform PCA dimensionality reduction
        
        Parameters:
        X (numpy.ndarray): Input features
        n_components (int): Desired number of components
        
        Returns:
        dict: Dictionary containing reduced features and evaluation metrics
        """
        # Standardize the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Fit and transform PCA
        pca = PCA(n_components=n_components)
        X_reduced = pca.fit_transform(X_scaled)
        
        # Get reconstructed data
        X_reconstructed = pca.inverse_transform(X_reduced)
        X_reconstructed = scaler.inverse_transform(X_reconstructed)
        
        # Calculate metrics
        reconstruction_mse = mean_squared_error(X, X_reconstructed)
        reconstruction_r2 = r2_score(X, X_reconstructed)
        
        return {
            'reduced_features': X_reduced,
            'reconstruction_mse': reconstruction_mse,
            'reconstruction_r2': reconstruction_r2,
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'cumulative_variance_ratio': np.cumsum(pca.explained_variance_ratio_),
            'components': pca.components_
        }

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

def evaluate_autoencoder(X, encoding_dim, epochs=100, batch_size=32):
    """
    Train and evaluate autoencoder for dimensionality reduction
    
    Parameters:
    X (numpy.ndarray): Input features
    y (numpy.ndarray): Target variables
    encoding_dim (int): Desired number of dimensions after reduction
    """
    # Standardize the input data

    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
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

def compute_tsne(X, n_components=2, perplexity=30, random_state=42):
    """
    Perform t-SNE dimensionality reduction
    
    Parameters:
    X (numpy.ndarray): Input features
    n_components (int): Desired number of components (usually 2 or 3)
    perplexity (float): t-SNE perplexity parameter
    
    Returns:
    dict: Dictionary containing reduced features and metadata
    """
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit and transform t-SNE
    tsne = TSNE(n_components=n_components, 
                    perplexity=perplexity, 
                    random_state=random_state)
    X_reduced = tsne.fit_transform(X_scaled)
    
    # Calculate pairwise distances in original and reduced space
    original_distances = calculate_pairwise_distances(X_scaled)
    reduced_distances = calculate_pairwise_distances(X_reduced)
    
    # Calculate correlation between distance matrices
    distance_correlation = np.corrcoef(original_distances.flatten(), 
                                        reduced_distances.flatten())[0, 1]
    
    return {
        'reduced_features': X_reduced,
        'distance_correlation': distance_correlation
    }

def calculate_pairwise_distances( X):
    """Calculate pairwise Euclidean distances between points"""
    return np.sqrt(np.sum((X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2, axis=2))


d        = DATA().adds(csv(the.train))
l = len(d.rows[0]) - len(d.cols.y)
X = np.array(d.rows)
X = X[:, :l]

print(compute_tsne(X, 3))