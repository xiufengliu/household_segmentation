"""
Deep clustering methods for household energy segmentation.

This module implements various deep learning-based clustering approaches
including Deep Embedded Clustering (DEC) and weather-fused variants.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import KLDivergence
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from typing import Optional, Tuple, Union

from .base import BaseDeepClusteringMethod, BaseMultiModalClusteringMethod
from ..models.clustering_layers import ClusteringLayer
from ..utils.logging import LoggerMixin


class DeepEmbeddedClustering(BaseDeepClusteringMethod):
    """
    Deep Embedded Clustering (DEC) implementation.
    
    This class implements the DEC algorithm which jointly learns
    feature representations and cluster assignments.
    """
    
    def __init__(self,
                 n_clusters: int,
                 embedding_dim: int = 10,
                 alpha: float = 1.0,
                 random_state: Optional[int] = None):
        """
        Initialize DEC model.
        
        Args:
            n_clusters: Number of clusters
            embedding_dim: Dimension of learned embeddings
            alpha: Degrees of freedom for Student's t-distribution
            random_state: Random state for reproducibility
        """
        super().__init__(n_clusters, embedding_dim, random_state)
        self.alpha = alpha
        self.clustering_model_ = None
        self.history_ = None
        
    def initialize_with_autoencoder(self, autoencoder):
        """
        Initialize DEC with a pre-trained autoencoder.
        
        Args:
            autoencoder: Pre-trained autoencoder model
        """
        self.encoder_ = autoencoder.encoder
        self.log_info("DEC initialized with pre-trained autoencoder")
        
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None,
            epochs: int = 30, batch_size: int = 32, 
            tolerance: float = 0.001, update_interval: int = 1,
            verbose: int = 1) -> 'DeepEmbeddedClustering':
        """
        Fit the DEC model.
        
        Args:
            X: Input data
            y: Ignored, present for API consistency
            epochs: Number of training epochs
            batch_size: Batch size for training
            tolerance: Convergence tolerance
            update_interval: Interval for updating target distribution
            verbose: Verbosity level
            
        Returns:
            self: Fitted DEC object
        """
        if self.encoder_ is None:
            raise ValueError("DEC must be initialized with an autoencoder first")
            
        self.log_info(f"Training DEC for {epochs} epochs...")
        
        # Generate initial embeddings
        embeddings = self.encoder_.predict(X, verbose=0)
        
        # Initialize cluster centroids with K-means
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init='auto')
        kmeans.fit(embeddings)
        initial_centroids = kmeans.cluster_centers_.astype(np.float32)
        
        # Build clustering model
        self._build_clustering_model(X.shape[1:])
        
        # Set initial cluster centroids
        clustering_layer = self.clustering_model_.layers[-1]
        clustering_layer.set_weights([initial_centroids])
        
        # Compile model
        self.clustering_model_.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=KLDivergence()
        )
        
        # Training loop
        y_pred_last = np.zeros(len(X), dtype=int)
        
        for epoch in range(epochs):
            # Update target distribution P
            if epoch % update_interval == 0:
                q = self.clustering_model_.predict(X, verbose=0)
                p = self._target_distribution(q)
                
                # Check for convergence
                y_pred = np.argmax(q, axis=1)
                delta_label = np.sum(y_pred != y_pred_last).astype(float) / len(y_pred)
                y_pred_last = y_pred
                
                if verbose:
                    self.log_info(f"Epoch {epoch+1}/{epochs} - Labels changed: {delta_label:.4f}")
                    
                if epoch > 0 and delta_label < tolerance:
                    self.log_info(f"Converged at epoch {epoch+1}")
                    break
            
            # Shuffle data
            X_shuffled, p_shuffled = shuffle(X, p, random_state=epoch)
            
            # Train on batches
            n_batches = int(np.ceil(len(X) / batch_size))
            epoch_loss = []
            
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(X))
                
                batch_X = X_shuffled[start_idx:end_idx]
                batch_p = p_shuffled[start_idx:end_idx]
                
                loss = self.clustering_model_.train_on_batch(batch_X, batch_p)
                epoch_loss.append(loss)
                
            if verbose:
                avg_loss = np.mean(epoch_loss)
                self.log_info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f}")
        
        # Final predictions
        self.labels_ = self.predict(X)
        self.is_fitted_ = True
        
        self.log_info("DEC training completed")
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict cluster labels for new data.
        
        Args:
            X: Input data
            
        Returns:
            Cluster labels
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before prediction")
            
        q = self.clustering_model_.predict(X, verbose=0)
        return np.argmax(q, axis=1)
        
    def get_embeddings(self, X: np.ndarray) -> np.ndarray:
        """
        Get learned embeddings for input data.
        
        Args:
            X: Input data
            
        Returns:
            Learned embeddings
        """
        if self.encoder_ is None:
            raise ValueError("Encoder not available")
            
        return self.encoder_.predict(X, verbose=0)
        
    def _build_clustering_model(self, input_shape: Tuple[int, ...]) -> None:
        """Build the clustering model."""
        input_layer = tf.keras.layers.Input(shape=input_shape)
        embeddings = self.encoder_(input_layer)
        
        clustering_layer = ClusteringLayer(
            n_clusters=self.n_clusters,
            alpha=self.alpha,
            name='clustering_layer'
        )
        
        cluster_probs = clustering_layer(embeddings)
        
        self.clustering_model_ = Model(
            inputs=input_layer,
            outputs=cluster_probs,
            name='dec_model'
        )
        
    def _target_distribution(self, q: np.ndarray) -> np.ndarray:
        """
        Compute target distribution P from current soft assignments Q.
        
        Args:
            q: Current soft assignments
            
        Returns:
            Target distribution P
        """
        weight = q ** 2 / np.sum(q, axis=0)
        return (weight.T / np.sum(weight, axis=1)).T


class WeatherFusedDEC(BaseMultiModalClusteringMethod):
    """
    Weather-fused Deep Embedded Clustering.
    
    This class extends DEC to incorporate weather information using
    multi-modal fusion and attention mechanisms.
    """
    
    def __init__(self,
                 n_clusters: int,
                 embedding_dim: int = 10,
                 alpha: float = 1.0,
                 random_state: Optional[int] = None):
        """
        Initialize Weather-fused DEC model.
        
        Args:
            n_clusters: Number of clusters
            embedding_dim: Dimension of learned embeddings
            alpha: Degrees of freedom for Student's t-distribution
            random_state: Random state for reproducibility
        """
        super().__init__(n_clusters, embedding_dim, random_state)
        self.alpha = alpha
        self.clustering_model_ = None
        self.history_ = None
        
    def initialize_with_autoencoder(self, weather_autoencoder):
        """
        Initialize with a pre-trained weather-fused autoencoder.
        
        Args:
            weather_autoencoder: Pre-trained weather-fused autoencoder
        """
        self.encoder_ = weather_autoencoder.encoder
        self.log_info("Weather-fused DEC initialized with pre-trained autoencoder")
        
    def fit(self, X_primary: np.ndarray, X_secondary: Optional[np.ndarray] = None,
            y: Optional[np.ndarray] = None, epochs: int = 30, batch_size: int = 32,
            tolerance: float = 0.001, update_interval: int = 1,
            verbose: int = 1) -> 'WeatherFusedDEC':
        """
        Fit the Weather-fused DEC model.
        
        Args:
            X_primary: Primary input data (load shapes)
            X_secondary: Secondary input data (weather data)
            y: Ignored, present for API consistency
            epochs: Number of training epochs
            batch_size: Batch size for training
            tolerance: Convergence tolerance
            update_interval: Interval for updating target distribution
            verbose: Verbosity level
            
        Returns:
            self: Fitted Weather-fused DEC object
        """
        if self.encoder_ is None:
            raise ValueError("Weather-fused DEC must be initialized with an autoencoder first")
            
        if X_secondary is None:
            raise ValueError("Weather data (X_secondary) is required for weather-fused DEC")
            
        self.log_info(f"Training Weather-fused DEC for {epochs} epochs...")
        
        # Generate initial embeddings
        embeddings = self.encoder_.predict([X_primary, X_secondary], verbose=0)
        
        # Initialize cluster centroids with K-means
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init='auto')
        kmeans.fit(embeddings)
        initial_centroids = kmeans.cluster_centers_.astype(np.float32)
        
        # Build clustering model
        self._build_clustering_model(X_primary.shape[1:], X_secondary.shape[1:])
        
        # Set initial cluster centroids
        clustering_layer = self.clustering_model_.layers[-1]
        clustering_layer.set_weights([initial_centroids])
        
        # Compile model
        self.clustering_model_.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=KLDivergence()
        )
        
        # Training loop
        y_pred_last = np.zeros(len(X_primary), dtype=int)
        
        for epoch in range(epochs):
            # Update target distribution P
            if epoch % update_interval == 0:
                q = self.clustering_model_.predict([X_primary, X_secondary], verbose=0)
                p = self._target_distribution(q)
                
                # Check for convergence
                y_pred = np.argmax(q, axis=1)
                delta_label = np.sum(y_pred != y_pred_last).astype(float) / len(y_pred)
                y_pred_last = y_pred
                
                if verbose:
                    self.log_info(f"Epoch {epoch+1}/{epochs} - Labels changed: {delta_label:.4f}")
                    
                if epoch > 0 and delta_label < tolerance:
                    self.log_info(f"Converged at epoch {epoch+1}")
                    break
            
            # Shuffle data
            X_primary_shuffled, X_secondary_shuffled, p_shuffled = shuffle(
                X_primary, X_secondary, p, random_state=epoch
            )
            
            # Train on batches
            n_batches = int(np.ceil(len(X_primary) / batch_size))
            epoch_loss = []
            
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(X_primary))
                
                batch_X_primary = X_primary_shuffled[start_idx:end_idx]
                batch_X_secondary = X_secondary_shuffled[start_idx:end_idx]
                batch_p = p_shuffled[start_idx:end_idx]
                
                loss = self.clustering_model_.train_on_batch(
                    [batch_X_primary, batch_X_secondary], 
                    batch_p
                )
                epoch_loss.append(loss)
                
            if verbose:
                avg_loss = np.mean(epoch_loss)
                self.log_info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f}")
        
        # Final predictions
        self.labels_ = self.predict(X_primary, X_secondary)
        self.is_fitted_ = True
        
        self.log_info("Weather-fused DEC training completed")
        return self
        
    def predict(self, X_primary: np.ndarray, 
                X_secondary: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Predict cluster labels for new data.
        
        Args:
            X_primary: Primary input data
            X_secondary: Secondary input data
            
        Returns:
            Cluster labels
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before prediction")
            
        if X_secondary is None:
            raise ValueError("Weather data is required for prediction")
            
        q = self.clustering_model_.predict([X_primary, X_secondary], verbose=0)
        return np.argmax(q, axis=1)
        
    def get_embeddings(self, X_primary: np.ndarray, 
                      X_secondary: np.ndarray) -> np.ndarray:
        """
        Get learned embeddings for input data.
        
        Args:
            X_primary: Primary input data
            X_secondary: Secondary input data
            
        Returns:
            Learned embeddings
        """
        if self.encoder_ is None:
            raise ValueError("Encoder not available")
            
        return self.encoder_.predict([X_primary, X_secondary], verbose=0)
        
    def _build_clustering_model(self, 
                               primary_input_shape: Tuple[int, ...],
                               secondary_input_shape: Tuple[int, ...]) -> None:
        """Build the weather-fused clustering model."""
        primary_input = tf.keras.layers.Input(shape=primary_input_shape, name='primary_input')
        secondary_input = tf.keras.layers.Input(shape=secondary_input_shape, name='secondary_input')
        
        embeddings = self.encoder_([primary_input, secondary_input])
        
        clustering_layer = ClusteringLayer(
            n_clusters=self.n_clusters,
            alpha=self.alpha,
            name='weather_clustering_layer'
        )
        
        cluster_probs = clustering_layer(embeddings)
        
        self.clustering_model_ = Model(
            inputs=[primary_input, secondary_input],
            outputs=cluster_probs,
            name='weather_fused_dec_model'
        )
        
    def _target_distribution(self, q: np.ndarray) -> np.ndarray:
        """
        Compute target distribution P from current soft assignments Q.
        
        Args:
            q: Current soft assignments
            
        Returns:
            Target distribution P
        """
        weight = q ** 2 / np.sum(q, axis=0)
        return (weight.T / np.sum(weight, axis=1)).T
