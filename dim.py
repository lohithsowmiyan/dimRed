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
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score
from config import parse_arguments


class DimensionalityReduction:

    def __init__(self, X):
        self.scaler = StandardScaler()
        self.X = X
    

    def compute_PCA(self, n_components):
            """
            Perform PCA dimensionality reduction
            
            Parameters:
            X (numpy.ndarray): Input features
            n_components (int): Desired number of components
            
            Returns:
            dict: Dictionary containing reduced features and evaluation metrics
            """
            # Standardize the data
         
            X_scaled = self.scaler.fit_transform(self.X)
            
            # Fit and transform PCA
            pca = PCA(n_components=n_components)
            X_reduced = pca.fit_transform(X_scaled)
            
            # Get reconstructed data
            X_reconstructed = pca.inverse_transform(X_reduced)
            X_reconstructed = self.scaler.inverse_transform(X_reconstructed)
            
            # Calculate metrics
            reconstruction_mse = mean_squared_error(self.X, X_reconstructed)
            reconstruction_r2 = r2_score(self.X, X_reconstructed)
            
            # return {
            #     'reduced_features': X_reduced,
            #     'reconstruction_mse': reconstruction_mse,
            #     'reconstruction_r2': reconstruction_r2,
            #     'explained_variance_ratio': pca.explained_variance_ratio_,
            #     'cumulative_variance_ratio': np.cumsum(pca.explained_variance_ratio_),
            #     'components': pca.components_
            # }

            return X_reduced

    def _create_autoencoder(self, input_dim, encoding_dim):
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

    def evaluate_autoencoder(self, encoding_dim, epochs=100, batch_size=32):
        """
        Train and evaluate autoencoder for dimensionality reduction
        
        Parameters:
        X (numpy.ndarray): Input features
        y (numpy.ndarray): Target variables
        encoding_dim (int): Desired number of dimensions after reduction
        """
        # Standardize the input data

        
      
        X_scaled = self.scaler.fit_transform(self.X)
        
        # Split the data
        X_train, X_test = train_test_split(X_scaled, test_size=0.2, random_state=42)
        
        # Create and train autoencoder
        autoencoder, encoder = self._create_autoencoder(self.X.shape[1], encoding_dim)
        autoencoder.fit(X_train, X_train, 
                    epochs=epochs, 
                    batch_size=batch_size, 
                    validation_data=(X_test, X_test),
                    verbose=0)
        
        # Get encoded (reduced) representations
        X_encoded = encoder.predict(X_scaled)
        
        # Get reconstructed data
        X_reconstructed = autoencoder.predict(X_scaled)
        X_reconstructed = self.scaler.inverse_transform(X_reconstructed)
        
        # Calculate evaluation metrics
        reconstruction_mse = mean_squared_error(self.X, X_reconstructed)
        reconstruction_r2 = r2_score(self.X, X_reconstructed)
        
        # Calculate explained variance ratio for encoded features
        original_var = np.var(self.X, axis=0).sum()
        encoded_var = np.var(X_encoded, axis=0).sum()
        explained_variance_ratio = encoded_var / original_var
        
        # return {
        #     'encoded_features': X_encoded,
        #     'reconstruction_mse': reconstruction_mse,
        #     'reconstruction_r2': reconstruction_r2,
        #     'explained_variance_ratio': explained_variance_ratio,
        #     'encoder': encoder,
        #     'autoencoder': autoencoder
        # }

        return X_encoded

    def compute_tsne(self, n_components=2, perplexity=30, random_state=42):
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
        
        X_scaled = self.scaler.fit_transform(self.X)
        
        # Fit and transform t-SNE
        tsne = TSNE(n_components=n_components, 
                        perplexity=perplexity, 
                        random_state=random_state)
        X_reduced = tsne.fit_transform(X_scaled)
        
        # Calculate pairwise distances in original and reduced space
        original_distances = self._calculate_pairwise_distances(X_scaled)
        reduced_distances = self._calculate_pairwise_distances(X_reduced)
        
        # Calculate correlation between distance matrices
        distance_correlation = np.corrcoef(original_distances.flatten(), 
                                            reduced_distances.flatten())[0, 1]
        
        # return {
        #     'reduced_features': X_reduced,
        #     'distance_correlation': distance_correlation
        # }

        return X_reduced

    def _calculate_pairwise_distances(self, X):
        """Calculate pairwise Euclidean distances between points"""
        return np.sqrt(np.sum((X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2, axis=2))



class Prediction:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        if len(y.shape) == 1:
            self.y = y.reshape(-1, 1)

        regressor = RandomForestRegressor(random_state=42)

        # Train the model
        if self.y.shape[1] > 1:
            self.model = MultiOutputRegressor(regressor)
        else:
            self.model = regressor

    def evaluate(self):

        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, shuffle = True)

        self.model.fit(X_train, y_train)

        # Predict and evaluate
        y_pred = self.model.predict(X_test)
        scores = [r2_score(y_test[:, i], y_pred[:, i]) for i in range(self.y.shape[1])]

        return sum(scores) / len(scores)

class Optimization:
    def __init__(self, X, y):
        self.X = X.tolist()
        self.y = y

        self.scores = {
            'explore' : lambda B,R : B + R / abs(B - R) + 1e-7,
            'exploit' : lambda B,R : B - R
        }

    def evaluate(self, score = 'exploit', labels = 30):

        self.X = [["Col" + str(i) for i in range(len(self.X[0]))]] + self.X

        hstack_2d_lists = lambda lists: [sum(row, []) for row in zip(*lists)]
        cur_d = hstack_2d_lists([self.X, self.y])
        

        d = DATA().adds(cur_d)

        activation = self.scores[score]


        tmp=d.shuffle().activeLearning(score=activation)
        rnd      = lambda z: ((int(z/dull) * dull) if the.Dull else z)
        result = rnd(d.chebyshev(tmp[0]))
        return [result]
        


metrics = {
    'predict' : Prediction,
    'optimize' : Optimization
}

def dim_exp():

    args = parse_arguments()

    metric = metrics[args.metric]
    
    d        = DATA().adds(csv(args.dataset))
    l = len(d.rows[0]) - len(d.cols.y)
    X = np.array(d.rows)
    X_new = X[:, :l]
    y = X[:, l:]


    y_cols = [[col.txt for col in d.cols.y]] 
    y = y_cols +  y.tolist()

    repeats = 20
    dims = [2,3,4,5]

    somes = []
    result = []

    for _ in range(repeats):
            
        perf = metric(X_new, y)
        result += perf.evaluate()

    pre=f"original"
    tag = pre
    somes +=   [stats.SOME(result,    tag)]

    for dim in dims:
        dimRed = DimensionalityReduction(X_new)
        X_transformed = dimRed.compute_PCA(dim)
        result = []
        for _ in range(repeats):
            
            perf = metric(X_transformed, y)
            result += perf.evaluate()

        pre=f"PCA/{dim}"
        tag = pre
        somes +=   [stats.SOME(result,    tag)]

    for dim in dims:
        dimRed = DimensionalityReduction(X_new)
        X_transformed = dimRed.evaluate_autoencoder(dim)
        result = []
        for _ in range(repeats):
            
            perf = metric(X_transformed, y)
            result += perf.evaluate()

        pre=f"autoencoder/{dim}"
        tag = pre
        somes +=   [stats.SOME(result,    tag)]

    stats.report(somes, 0.01)

dim_exp()