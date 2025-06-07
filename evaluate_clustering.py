import tensorflow as tf
from tensorflow.keras.models import load_model, Model
import numpy as np
import os
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
# For t-SNE (optional, if we want to show a quick plot if possible in environment)
# from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt

# Ensure custom layer is available
@tf.keras.utils.register_keras_serializable(package='Custom', name='ClusteringLayer')
class ClusteringLayer(tf.keras.layers.Layer):
    def __init__(self, n_clusters, name='clustering_layer', alpha=1.0, **kwargs):
        super(ClusteringLayer, self).__init__(name=name, **kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
    def build(self, input_shape):
        if len(input_shape) != 2: # Should be (None, embedding_dim)
             raise ValueError(f"Expected 2D input (batch_size, features), got input_shape={input_shape}")
        self.clusters = self.add_weight(name='cluster_centroids', shape=(self.n_clusters, input_shape[-1]), initializer='glorot_uniform', trainable=True)
        super(ClusteringLayer, self).build(input_shape)
    def call(self, inputs):
        q = tf.expand_dims(inputs, axis=1) - tf.expand_dims(self.clusters, axis=0)
        q = tf.reduce_sum(tf.square(q), axis=2) / self.alpha
        q = 1.0 + q
        q = tf.pow(q, -(self.alpha + 1.0) / 2.0)
        return q / tf.reduce_sum(q, axis=1, keepdims=True)
    def get_config(self):
        config = super(ClusteringLayer, self).get_config()
        config.update({'n_clusters': self.n_clusters, 'alpha': self.alpha})
        return config

print(f"TensorFlow version: {tf.__version__}")

# --- Parameters ---
num_samples = 100
timesteps = 24
features = 1
trained_model_path = 'trained_clustering_model.keras'

# --- 2. Load Scaled Dummy Data ---
np.random.seed(42) # Use the same seed as training to get the same data
dummy_data = np.random.rand(num_samples, timesteps, features).astype(np.float32)
for i in range(num_samples): # Add some structure
    peak_time = np.random.randint(5, 19)
    peak_val = np.random.uniform(0.5, 1.0)
    dummy_data[i, peak_time-2:peak_time+3, 0] = peak_val * np.array([0.5, 0.8, 1.0, 0.8, 0.5])
    dummy_data[i] += 0.1 * np.random.randn(timesteps, features).astype(np.float32)
min_val, max_val = np.min(dummy_data), np.max(dummy_data)
scaled_data = (dummy_data - min_val) / (max_val - min_val) if max_val > min_val else np.zeros_like(dummy_data)
if features == 1 and len(scaled_data.shape) == 2: # Ensure (samples, timesteps, 1)
    scaled_data = np.expand_dims(scaled_data, axis=-1)
print(f"Generated and scaled dummy data with shape: {scaled_data.shape}")

# --- 3. Load Trained Clustering Model ---
if not os.path.exists(trained_model_path):
    print(f"ERROR: Trained clustering model file not found at {trained_model_path}")
    # Fallback: create and save a dummy trained model
    print(f"Attempting to create a dummy trained model as {trained_model_path} was not found.")
    _input_enc = tf.keras.layers.Input(shape=(timesteps, features), name="input_layer")
    _x_enc = tf.keras.layers.Conv1D(16, 3, activation='relu', padding='same')(_input_enc)
    _x_enc = tf.keras.layers.MaxPooling1D(2, padding='same')(_x_enc)
    _x_enc = tf.keras.layers.Conv1D(8, 3, activation='relu', padding='same')(_x_enc)
    _x_enc = tf.keras.layers.MaxPooling1D(2, padding='same')(_x_enc)
    _x_enc = tf.keras.layers.Flatten()(_x_enc)
    _embs = tf.keras.layers.Dense(10, activation='relu', name='embedding')(_x_enc) # Assuming embedding_dim=10

    _clustering_layer_inst = ClusteringLayer(n_clusters=3, name='clustering_probs_dummy_eval') # n_clusters=3 example
    _clustering_out = _clustering_layer_inst(_embs)
    _dummy_clustering_model = Model(inputs=_input_enc, outputs=_clustering_out, name='clustering_model_dummy_eval')

    # Initialize weights for dummy clustering layer (randomly)
    # Build the layer by calling it once
    _dummy_clustering_model.predict(scaled_data[:1], verbose=0) # Call predict to build
    _clustering_layer_inst.set_weights([np.random.rand(3, 10).astype(np.float32)]) # n_clusters=3, embedding_dim=10

    _dummy_clustering_model.save(trained_model_path)
    print(f"Dummy trained model created and saved to {trained_model_path}. This indicates an issue in the overall workflow.")

print(f"Loading trained clustering model from {trained_model_path}...")
clustering_model = load_model(trained_model_path)
print("Trained clustering model loaded successfully.")

# --- 4. Extract Encoder and Get Embeddings ---
# The clustering_model is Model(inputs=encoder.input, outputs=clustering_layer(encoder.output))
# The input to the ClusteringLayer (the last layer, index -1) is the embedding.
# The layer before the ClusteringLayer is the one that produces the embeddings.
# If the encoder was a single functional model layer in the clustering model: clustering_model.layers[0]
# If encoder layers were added sequentially before clustering layer: clustering_model.layers[-2].output
if not clustering_model.layers:
    print("ERROR: Loaded clustering model has no layers.")
    exit()
if len(clustering_model.layers) < 2:
    print(f"ERROR: Loaded clustering model has only {len(clustering_model.layers)} layer(s), cannot identify encoder output.")
    exit()

# The input to the last layer (ClusteringLayer) is the output of the encoder part.
encoder_output_tensor = clustering_model.layers[-1].input
encoder_input_tensor = clustering_model.input
encoder_model = Model(inputs=encoder_input_tensor, outputs=encoder_output_tensor, name="encoder_from_clustering_model")

print("\n--- Extracted Encoder Summary ---")
encoder_model.summary()
embeddings = encoder_model.predict(scaled_data, verbose=0)
print(f"Embeddings shape: {embeddings.shape}")

# --- 5. Get Final Cluster Assignments ---
q_values = clustering_model.predict(scaled_data, verbose=0)
hard_assignments = np.argmax(q_values, axis=1)
print(f"Hard cluster assignments shape: {hard_assignments.shape}")
print(f"Unique clusters assigned: {np.unique(hard_assignments)}")

# Ensure there's more than 1 cluster for metrics, otherwise they fail
if len(np.unique(hard_assignments)) < 2:
    print("Only one cluster found. Skipping Silhouette, Davies-Bouldin, Calinski-Harabasz scores as they require at least 2 clusters.")
else:
    # --- 6. Calculate Internal Validation Metrics ---
    print("\n--- Internal Validation Metrics (on Embeddings) ---")

    silhouette = silhouette_score(embeddings, hard_assignments)
    print(f"Silhouette Score: {silhouette:.4f} (Higher is better, range -1 to 1)")

    davies_bouldin = davies_bouldin_score(embeddings, hard_assignments)
    print(f"Davies-Bouldin Index: {davies_bouldin:.4f} (Lower is better, minimum 0)")

    calinski_harabasz = calinski_harabasz_score(embeddings, hard_assignments)
    print(f"Calinski-Harabasz Index: {calinski_harabasz:.4f} (Higher is better)")

# --- 8. Briefly discuss qualitative visualization ---
print("\n--- Qualitative Analysis Discussion ---")
print("For qualitative analysis with real data, we would:")
print("1. Visualize average load shapes per cluster: Decode centroids or average cluster member series.")
print("2. Use t-SNE or UMAP to project embeddings into 2D/3D and color by cluster.")
print("   Example (conceptual):")
print("   # from sklearn.manifold import TSNE")
print("   # import matplotlib.pyplot as plt")
print("   # tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))")
print("   # embeddings_2d = tsne.fit_transform(embeddings)")
print("   # plt.figure(figsize=(10, 8))")
print("   # for cluster_id in np.unique(hard_assignments):")
print("   #     plt.scatter(embeddings_2d[hard_assignments==cluster_id, 0], embeddings_2d[hard_assignments==cluster_id, 1], label=f'Cluster {cluster_id}')")
print("   # plt.title('t-SNE visualization of learned embeddings')")
print("   # plt.xlabel('t-SNE feature 1')")
print("   # plt.ylabel('t-SNE feature 2')")
print("   # plt.legend()")
print("   # plt.show() # This would require a graphical backend if run in the subtask.")
