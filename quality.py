from ezr import * 
import scipy.stats as stats
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from scipy.stats import kurtosis, skew
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.ensemble import IsolationForest

def fractal_dimensions(dataset):
    # Fractal dimensions using box-counting method
    d = DATA().adds(csv(dataset))
    sr = len(d.rows[0]) - len(d.cols.y)
    data = np.array(d.rows)
    data = data[:, :sr]
    def box_counting(data, box_size):
        min_bounds = np.min(data, axis=0)
        max_bounds = np.max(data, axis=0)
        grid = np.ceil((max_bounds - min_bounds) / box_size).astype(int)
        occupied_boxes = set()

        for point in data:
            box_index = tuple(((point - min_bounds) // box_size).astype(int))
            occupied_boxes.add(box_index)

        return len(occupied_boxes)

    box_sizes = np.logspace(-1, 1, num=10)  # Box sizes ranging from 0.1 to 10
    box_counts = [box_counting(data, box_size) for box_size in box_sizes]
    log_box_sizes = np.log(box_sizes)
    log_box_counts = np.log(box_counts)

    fractal_dimension, _ = np.polyfit(log_box_sizes, log_box_counts, 1)

    return fractal_dimension

def data_spread_stats(dataset):
    d = DATA().adds(csv(dataset))
    sr = len(d.rows[0]) - len(d.cols.y)
    data = np.array(d.rows)
    data = data[:, :sr]
    covariance_matrix = np.cov(data, rowvar=False)
    eigenvalues = np.linalg.eigvalsh(covariance_matrix)  # Eigenvalues of the covariance matrix
    eigenvalues_sorted = np.sort(eigenvalues)[::-1]  # Sort in descending order

    return eigenvalues_sorted

def data_distribution_stats(dataset, distance_metric="euclidean", n_neighbors=5):
    d = DATA().adds(csv(dataset))
    sr = len(d.rows[0]) - len(d.cols.y)
    data = np.array(d.rows)
    data = data[:, :sr]
    # Smoothness: Average distances to the n_neighbors nearest points
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1, metric=distance_metric).fit(data)
    distances, _ = nbrs.kneighbors(data)
    smoothness = np.mean(distances[:, 1:], axis=1)  # Exclude self-distance
    
    pairwise_distances = squareform(pdist(data, metric=distance_metric)) # Density-based metrics: Mean and standard deviation of local densities
    density = 1 / (np.mean(distances[:, 1:], axis=1) + 1e-10)  # Avoid division by zero
    density_mean = np.mean(density)
    density_std = np.std(density)

    return {
        'Density Mean' : density_mean,
        'Density Std' : density_std
    }




def landscape_analysis(dataset, distance_metric="euclidean", n_neighbors=5):
    """
    Perform landscape analysis to measure smoothness and variability in the feature space.

    Parameters:
    - data: np.ndarray, shape (n_samples, n_features)
        Dataset where rows are samples and columns are features.
    - distance_metric: str, optional (default="euclidean")
        Distance metric to use (e.g., "euclidean", "manhattan", "cosine").
    - n_neighbors: int, optional (default=5)
        Number of neighbors to compute local smoothness.

    Returns:
    - results: dict
        Dictionary containing landscape metrics (smoothness, variability).
    """
    d = DATA().adds(csv(dataset))
    sr = len(d.rows[0]) - len(d.cols.y)
    print(dataset)
    print(sr)
    data = np.array(d.rows)

    data = data[:, :sr]

    # Pairwise distances between all points
    pairwise_distances = squareform(pdist(data, metric=distance_metric))

    # Smoothness: Average distances to the n_neighbors nearest points
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1, metric=distance_metric).fit(data)
    distances, _ = nbrs.kneighbors(data)
    smoothness = np.mean(distances[:, 1:], axis=1)  # Exclude self-distance

    # Variability: Entropy of the pairwise distance distribution
    flattened_distances = pairwise_distances[np.triu_indices_from(pairwise_distances, k=1)]
    hist, _ = np.histogram(flattened_distances, bins=50, density=True)
    variability = entropy(hist)

    # Assemble results
    results = {
        "Smoothness (mean distance to neighbors)": np.mean(smoothness),
        "Variability (distance distribution entropy)": variability,
    }

    return results

def feature_space_analysis(dataset):
    """
    Perform feature space analysis on a dataset.

    Parameters:
    - data: np.ndarray or pd.DataFrame
        Dataset where rows are samples and columns are features.

    Returns:
    - results: dict
        Dictionary containing feature space metrics.
    """
    d = DATA().adds(csv(dataset))
    sr = len(d.rows[0]) - len(d.cols.y)
    print(dataset)
    print(sr)
    data = np.array(d.rows)

    data = data[:, :sr]


    # Compute covariance matrix
    covariance_matrix = np.cov(data, rowvar=False)

    # Compute explained variance using PCA
    pca = PCA()
    pca.fit(data)
    explained_variance = pca.explained_variance_ratio_

    # Compute kurtosis and skewness for each feature
    feature_kurtosis = kurtosis(data, axis=0, fisher=True)
    feature_skewness = skew(data, axis=0)

    # Compute feature correlation matrix
    correlation_matrix = np.corrcoef(data, rowvar=False)

    # Assemble results
    results = {
        "Covariance Matrix": covariance_matrix,
        "Explained Variance (PCA)": explained_variance,
        "Feature Kurtosis": feature_kurtosis,
        "Feature Skewness": feature_skewness,
        "Correlation Matrix": correlation_matrix,
    }

    return results

def safe_correlation_dimension(dataset, max_dim=10):
    """
    Robust intrinsic dimensionality estimation using correlation dimension.
    """
    d = DATA().adds(csv(dataset))
    sr = len(d.rows[0]) - len(d.cols.y)
    print(dataset)
    print(sr)
    data = np.array(d.rows)

    data = data[:, :sr]

    try:
        n_samples, n_features = data.shape
        
        # Compute pairwise distances with numerical stability
        distances = []
        for i in range(n_samples):
            for j in range(i+1, n_samples):
                dist = np.linalg.norm(data[i] - data[j])
                if dist > 0:  # Avoid zero distances
                    distances.append(dist)
        
        if not distances:
            logger.warning("No valid distances found. Check input data.")
            return None
        
        distances = np.sort(distances)
        
        # Robust log computation
        log_distances = np.log(distances + np.finfo(float).eps)
        
        correlation_dims = []
        for d in range(1, min(max_dim + 1, len(distances))):
            try:
                # Use percentile-based distance thresholds
                thresholds = np.percentile(distances, [10, 30, 50, 70, 90])
                log_counts = [np.log(np.sum(distances <= t) / len(distances) + np.finfo(float).eps) 
                               for t in thresholds]
                
                # Robust linear regression
                slope, _ = np.polyfit(log_distances[:len(log_counts)], log_counts, 1)
                correlation_dims.append(slope)
            except Exception as e:
                logger.error(f"Error in dimension estimation: {e}")
        
        if correlation_dims:
            return  correlation_dims
        else:
            logger.warning("Could not estimate intrinsic dimensions.")
            return None
    
    except Exception as e:
        logger.error(f"Correlation dimension computation failed: {e}")
        return None

def intrinsic_dimensionality(dataset = 'data/optimize/config/SS-X.csv', max_dim=5):
    """
    Estimate intrinsic dimensionality using correlation dimension.
    """
    d = DATA().adds(csv(dataset))
    sr = len(d.rows[0]) - len(d.cols.y)
    print(dataset)
    print(sr)
    data = np.array(d.rows)

    data = data[:, :sr]
    print(data.shape)
    

    #pca = PCA()
    #pca.fit(data)
    
    # Compute cumulative explained variance
    # cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    
    # # Find number of components explaining 95% of variance
    # pca_intrinsic_dim = np.argmax(cumulative_variance >= 0.95) + 1
    # results = {}
    # results['pca'] = {
    #     'intrinsic_dimension': pca_intrinsic_dim,
    #     'explained_variance': cumulative_variance
    #     }

    # return pca_intrinsic_dim

    # try:
    #         # Estimate intrinsic dimensionality using UMAP
    #         reducer = umap.UMAP(n_components=10, random_state=42)
    #         reducer.fit(data)
            
    #         results = reducer.n_components
            
    # except ImportError:
    #         print("UMAP method requires 'umap-learn' package to be installed.")
    
    # return results

    n_samples, n_features = data.shape
            
    # Compute pairwise distances
    distances = []
    for i in range(n_samples):
        for j in range(i+1, n_samples):
            distances.append(np.linalg.norm(data[i] - data[j]))
    
    distances = np.sort(distances)
    
    # Estimate scaling exponent
    log_distances = np.log(distances)
    correlation_dims = []
    
    for d in range(1, max_dim + 1):
        # Compute correlation integral
        log_counts = []
        for eps in np.percentile(distances, [10, 30, 50, 70, 90]):
            count = np.sum(distances <= eps)
            log_counts.append(np.log(count / len(distances)))
        
        # Linear regression to estimate scaling exponent
        slope, _ = np.polyfit(log_distances[:len(log_counts)], log_counts, 1)
        correlation_dims.append(slope)
    
    return np.median(correlation_dims)



import os

def list_files_in_directory(directory):
    csv_files = []
    try:
        # Walk through the directory
        for root, dirs, files in os.walk(directory):
            for file in files:
                # Check if the file has a `.csv` extension
                if file.endswith(".csv"):
                    # Append the full path of the CSV file
                    csv_files.append(os.path.join(root, file))
    except Exception as e:
        print(f"An error occurred: {e}")

    return csv_files

csv_files = list_files_in_directory('data/optimize/config')

# for file in csv_files:
#     print(f'filename:{file}, spread:{fractal_dimensions(file)}')



class Hardness:

    def __init__(self, dataset = 'data/optimize/misc/auto93.csv'):

        d = DATA().adds(csv(dataset))
        sr = len(d.rows[0]) - len(d.cols.y)
        data = np.array(d.rows)
        self.data = data[:, :sr]

    def isolation_hardness(self):
        """
        Compute instance hardness using Isolation Forest anomaly scores.

        Parameters:
        - data (ndarray or DataFrame): Feature matrix (instances × features).

        Returns:
        - instance_hardness_scores (ndarray): Anomaly scores for each instance.
        """
        # Train Isolation Forest
        iso_forest = IsolationForest(random_state=42, contamination=0.1)
        anomaly_scores = -iso_forest.fit_predict(self.data)

        # Higher anomaly score indicates harder instance
        return anomaly_scores

    def clustering_hardness(self, n_clusters=3):
        """
        Compute instance hardness using clustering distance to cluster centers.

        Parameters:
        - data (ndarray or DataFrame): Feature matrix (instances × features).
        - n_clusters (int): Number of clusters.

        Returns:
        - instance_hardness_scores (ndarray): Distance to the closest cluster center for each instance.
        """
        # Perform KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(self.data)

        # Find the closest cluster center for each instance
        _, distances = pairwise_distances_argmin_min(self.data, kmeans.cluster_centers_)

        return distances 

    def density_hardness(self, k=5, metric="euclidean"):
        """
        Compute instance hardness without target values using k-NN density.

        Parameters:
        - data (ndarray or DataFrame): Feature matrix (instances × features).
        - k (int): Number of neighbors to consider.
        - metric (str): Distance metric to use (default: 'euclidean').

        Returns:
        - instance_hardness_scores (ndarray): Instance hardness scores for each instance.
        """
        # Compute pairwise distances
        distances = pairwise_distances(self.data, metric=metric)

        # Sort distances to get k nearest neighbors (excluding self-distance at index 0)
        sorted_distances = np.sort(distances, axis=1)[:, 1 : k + 1]

        # Compute density as the average distance to k neighbors
        density = np.mean(sorted_distances, axis=1)

        # Higher density → less hard, lower density → more hard
        # Hardness is inverse of density
        instance_hardness_scores = 1 / (density + 1e-8)  # Add epsilon to avoid division by zero

        return instance_hardness_scores

# for file in csv_files:
#     obj = Hardness(file)
#     print("filename :",file)
#     print("isolation forest scores :", obj.isolation_hardness())
#     print("clustering scores :", obj.clustering_hardness())
#     print("density hardness :", obj.density_hardness())

d = DATA().adds(csv('data/optimize/hpo/healthCloseIsses12mths0001-hard.csv'))