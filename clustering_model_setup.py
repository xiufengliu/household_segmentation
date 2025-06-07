import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Layer, Input
import numpy as np
from sklearn.cluster import KMeans
import os

print(f"TensorFlow version: {tf.__version__}")

# --- Parameters ---
num_samples = 100
timesteps = 24
features = 1
embedding_dim = 10 # Should match the pre-trained encoder's output
n_clusters = 3 # Example number of clusters
pretrained_model_path = 'pretrained_autoencoder_full_model.keras'

# --- 2. Generate/Load Scaled Dummy Data ---
np.random.seed(42)
dummy_data = np.random.rand(num_samples, timesteps, features).astype(np.float32)
for i in range(num_samples): # Add some structure
    peak_time = np.random.randint(5, 19)
    peak_val = np.random.uniform(0.5, 1.0)
    dummy_data[i, peak_time-2:peak_time+3, 0] = peak_val * np.array([0.5, 0.8, 1.0, 0.8, 0.5])
    dummy_data[i] += 0.1 * np.random.randn(timesteps, features).astype(np.float32)

min_val = np.min(dummy_data)
max_val = np.max(dummy_data)
if max_val == min_val:
    scaled_data = np.zeros_like(dummy_data)
else:
    scaled_data = (dummy_data - min_val) / (max_val - min_val)
if features == 1 and len(scaled_data.shape) == 2: # Ensure (samples, timesteps, 1)
    scaled_data = np.expand_dims(scaled_data, axis=-1)
print(f"Generated and scaled dummy data with shape: {scaled_data.shape}")

# --- 3. Load Pre-trained Autoencoder and Extract Encoder ---
if not os.path.exists(pretrained_model_path):
    print(f"ERROR: Pre-trained model file not found at {pretrained_model_path}")
    # Fallback: create a dummy autoencoder and save it, then load.
    # This is to ensure the script can run end-to-end for testing even if a file is missing.
    print(f"Attempting to create a dummy pre-trained model as {pretrained_model_path} was not found.")
    _input_seq = tf.keras.layers.Input(shape=(timesteps, features), name="input_layer")
    _x = tf.keras.layers.Conv1D(16, 3, activation='relu', padding='same')(_input_seq)
    _x = tf.keras.layers.MaxPooling1D(2, padding='same')(_x)
    _x = tf.keras.layers.Conv1D(8, 3, activation='relu', padding='same')(_x)
    _x = tf.keras.layers.MaxPooling1D(2, padding='same')(_x)
    _shape_before_flatten = tf.keras.backend.int_shape(_x)[1:]
    _x = tf.keras.layers.Flatten()(_x)
    _encoded = tf.keras.layers.Dense(embedding_dim, activation='relu', name='embedding')(_x)
    _encoder_model = tf.keras.models.Model(_input_seq, _encoded, name="encoder")

    _latent_inputs = tf.keras.layers.Input(shape=(embedding_dim,), name='decoder_input')
    _x = tf.keras.layers.Dense(np.prod(_shape_before_flatten), activation='relu')(_latent_inputs)
    _x = tf.keras.layers.Reshape(_shape_before_flatten)(_x)
    _x = tf.keras.layers.UpSampling1D(2)(_x)
    _x = tf.keras.layers.Conv1D(8, 3, activation='relu', padding='same')(_x)
    _x = tf.keras.layers.UpSampling1D(2)(_x)
    _x = tf.keras.layers.Conv1D(16, 3, activation='relu', padding='same')(_x)
    _decoded = tf.keras.layers.Conv1D(features, 3, activation='sigmoid', padding='same')(_x)
    _decoder_model = tf.keras.models.Model(_latent_inputs, _decoded, name="decoder")

    _autoencoder_output = _decoder_model(_encoder_model(_input_seq))
    _autoencoder = tf.keras.models.Model(_input_seq, _autoencoder_output, name="autoencoder")
    _autoencoder.save(pretrained_model_path)
    print(f"Dummy pre-trained model saved to {pretrained_model_path}. This indicates an issue in the overall workflow if it happens.")


print(f"Loading pre-trained model from {pretrained_model_path}...")
autoencoder = load_model(pretrained_model_path) # Keras should load the model with its structure
encoder = autoencoder.get_layer('encoder') # This relies on the encoder part being named 'encoder'

if encoder is None:
    print("Could not find 'encoder' layer by name. Attempting to reconstruct from layers if 'embedding' layer exists.")
    try:
        encoder_output_tensor = autoencoder.get_layer('embedding').output
        encoder_input_tensor = autoencoder.input # Assuming autoencoder.input is the correct input for the encoder part
        encoder = Model(inputs=encoder_input_tensor, outputs=encoder_output_tensor, name='encoder_extracted')
    except AttributeError as e:
        print(f"Failed to extract encoder: {e}. The model structure might not be as expected.")
        exit()


print("Encoder extracted successfully.")
encoder.summary()

# --- 4. Define ClusteringLayer ---
@tf.keras.utils.register_keras_serializable()
class ClusteringLayer(Layer):
    def __init__(self, n_clusters, name='clustering_layer', alpha=1.0, **kwargs):
        super(ClusteringLayer, self).__init__(name=name, **kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha

    def build(self, input_shape):
        if len(input_shape) != 2:
            raise ValueError(f"Expected 2D input (batch_size, features), got input_shape={input_shape}")

        self.clusters = self.add_weight(
            name='cluster_centroids',
            shape=(self.n_clusters, input_shape[-1]),
            initializer='glorot_uniform',
            trainable=True
        )
        super(ClusteringLayer, self).build(input_shape)

    def call(self, inputs):
        q = tf.expand_dims(inputs, axis=1) - tf.expand_dims(self.clusters, axis=0)
        q = tf.reduce_sum(tf.square(q), axis=2)
        q = q / self.alpha
        q = 1.0 + q
        q = tf.pow(q, -(self.alpha + 1.0) / 2.0)
        q = q / tf.reduce_sum(q, axis=1, keepdims=True)
        return q

    def get_config(self):
        config = super(ClusteringLayer, self).get_config()
        config.update({'n_clusters': self.n_clusters, 'alpha': self.alpha})
        return config

# --- 5. Initialize Cluster Centroids ---
print("\nInitializing cluster centroids...")
embeddings = encoder.predict(scaled_data)
print(f"Embeddings shape: {embeddings.shape}")

kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
kmeans.fit(embeddings)
initial_centroids = kmeans.cluster_centers_.astype(np.float32) # Ensure float32 for Keras layer
print(f"Initial K-Means centroids shape: {initial_centroids.shape}, dtype: {initial_centroids.dtype}")


# --- 6. Construct the Clustering Model ---
inp = encoder.input
out_emb = encoder.output
clustering_layer_instance = ClusteringLayer(n_clusters=n_clusters, name='clustering_probs')
clustering_output = clustering_layer_instance(out_emb)

clustering_model = Model(inputs=inp, outputs=clustering_output, name='clustering_model')

# Set the initial weights for the clustering layer
clustering_layer_instance.set_weights([initial_centroids])
print("Set initial weights of ClusteringLayer with K-Means centroids.")

print("\n--- Clustering Model Summary ---")
clustering_model.summary()

# --- 7. Define KL Divergence Loss Function ---
kl_loss = tf.keras.losses.KLDivergence()
print(f"\nKL Divergence loss function defined: {kl_loss}")


# --- 8. Save the clustering model (optional) ---
clustering_model_path = 'clustering_model_definition.keras'
try:
    clustering_model.save(clustering_model_path) # Keras 3 should handle custom layers better
    print(f"Clustering model definition saved to {clustering_model_path}")
except Exception as e:
    print(f"Error saving clustering model: {e}")

# Verify loading the model with custom object
try:
    # With the decorator, custom_objects should not be strictly needed,
    # but it's good practice for clarity or if there are version/env issues.
    # However, Keras 3 aims to make this more seamless.
    loaded_clustering_model = load_model(clustering_model_path, compile=False)
    print("Successfully loaded clustering model (likely due to decorator).")
    loaded_clustering_model.summary()
except Exception as e:
    print(f"Error loading clustering model, even with decorator: {e}")
    print("Attempting to load with custom_objects explicitly:")
    try:
        loaded_clustering_model = load_model(clustering_model_path, custom_objects={'ClusteringLayer': ClusteringLayer}, compile=False)
        print("Successfully loaded clustering model with custom_objects.")
        loaded_clustering_model.summary()
    except Exception as e2:
        print(f"Error loading clustering model even with custom_objects: {e2}")
