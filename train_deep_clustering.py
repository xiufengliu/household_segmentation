import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.losses import KLDivergence
from tensorflow.keras.optimizers import Adam
import numpy as np
import os
from sklearn.utils import shuffle # For shuffling data and P together

# Ensure custom layer is available
@tf.keras.utils.register_keras_serializable(package='Custom', name='ClusteringLayer')
class ClusteringLayer(tf.keras.layers.Layer):
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

print(f"TensorFlow version: {tf.__version__}")

# --- Parameters ---
num_samples = 100
timesteps = 24
features = 1
n_clusters = 3 # Must match what was used when clustering_model_definition.keras was saved
clustering_model_path = 'clustering_model_definition.keras'
trained_model_save_path = 'trained_clustering_model.keras'

# Training parameters
epochs = 30
batch_size = 32
update_interval_P = 1 # Update P every epoch
tol = 0.001 # Convergence tolerance: 0.1% of labels change

# --- 2. Load Scaled Dummy Data ---
np.random.seed(42)
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

# --- 3. Load Clustering Model ---
if not os.path.exists(clustering_model_path):
    print(f"ERROR: Clustering model definition file not found at {clustering_model_path}")
    # Fallback: create a dummy clustering model if not found
    print(f"Attempting to create a dummy clustering model as {clustering_model_path} was not found.")
    _input_enc = tf.keras.layers.Input(shape=(timesteps, features), name="input_layer")
    _x_enc = tf.keras.layers.Conv1D(16, 3, activation='relu', padding='same')(_input_enc)
    _x_enc = tf.keras.layers.MaxPooling1D(2, padding='same')(_x_enc)
    _x_enc = tf.keras.layers.Conv1D(8, 3, activation='relu', padding='same')(_x_enc)
    _x_enc = tf.keras.layers.MaxPooling1D(2, padding='same')(_x_enc)
    _x_enc = tf.keras.layers.Flatten()(_x_enc)
    _embs = tf.keras.layers.Dense(10, activation='relu', name='embedding')(_x_enc) # Assuming embedding_dim=10
    _encoder_part = Model(inputs=_input_enc, outputs=_embs, name='encoder_dummy_part')

    _clustering_layer_inst = ClusteringLayer(n_clusters=n_clusters, name='clustering_probs_dummy')
    _clustering_out = _clustering_layer_inst(_embs)
    clustering_model = Model(inputs=_input_enc, outputs=_clustering_out, name='clustering_model_dummy')

    # Initialize weights for dummy clustering layer (randomly or from dummy kmeans)
    _dummy_embeddings = _encoder_part.predict(scaled_data, verbose=0)
    _kmeans_dummy = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto').fit(_dummy_embeddings)
    _clustering_layer_inst.set_weights([_kmeans_dummy.cluster_centers_.astype(np.float32)])

    clustering_model.save(clustering_model_path) # Save this dummy model
    print(f"Dummy clustering model created and saved to {clustering_model_path}. This indicates an issue in the overall workflow.")


print(f"Loading clustering model from {clustering_model_path}...")
clustering_model = load_model(clustering_model_path) # Decorator should handle custom layer
print("Clustering model loaded successfully.")
clustering_model.summary()

# --- 5. Compile Clustering Model ---
optimizer = Adam(learning_rate=0.001)
loss_fn = KLDivergence()
clustering_model.compile(optimizer=optimizer, loss=loss_fn)
print("Clustering model compiled with Adam optimizer and KLDivergence loss.")

# --- 6. Implement Custom Training Loop ---
print("\nStarting end-to-end training...")
y_pred_last = np.zeros(num_samples, dtype=int)
p_all = np.zeros((num_samples, n_clusters), dtype=np.float32) # Initialize p_all

for epoch in range(epochs):
    epoch_loss = []

    if epoch % update_interval_P == 0:
        print(f"Updating target distribution P for epoch {epoch+1}...")
        q_all = clustering_model.predict(scaled_data, verbose=0) # q_all is numpy array here

        # Convert q_all to tensor for tf operations
        q_all_tf = tf.convert_to_tensor(q_all, dtype=tf.float32)

        # Calculate P = Q^2 / sum(Q_j) normalized per sample
        # num = q_ij^2 / f_j where f_j = sum_i q_ij (column sums of q)
        # tf.reduce_sum(q_all_tf, axis=0) gives f_j for each cluster j
        num_p = q_all_tf**2 / tf.reduce_sum(q_all_tf, axis=0)
        # Then normalize num_p row-wise so that sum_j p_ij = 1
        p_all_tf = num_p / tf.reduce_sum(num_p, axis=1, keepdims=True)
        p_all = p_all_tf.numpy() # Convert final P to numpy for shuffle and batching

        y_pred_current = np.argmax(q_all, axis=1) # Use original q_all (numpy) for y_pred
        delta_label = np.sum(y_pred_current != y_pred_last).astype(float) / y_pred_current.shape[0]
        y_pred_last = y_pred_current

        print(f"Fraction of labels changed since last P update: {delta_label:.4f}")
        if epoch > 0 and delta_label < tol:
            print(f"Converged at epoch {epoch+1}. Change in labels ({delta_label:.4f}) < tolerance ({tol}).")
            break
        print(f"Target distribution P updated. Shape: {p_all.shape}")

    shuffled_data, shuffled_p = shuffle(scaled_data, p_all, random_state=epoch)

    num_batches = int(np.ceil(num_samples / batch_size))
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, num_samples)

        batch_x = shuffled_data[start_idx:end_idx]
        batch_p = shuffled_p[start_idx:end_idx]

        if batch_x.shape[0] == 0: continue # Skip empty batches

        current_loss = clustering_model.train_on_batch(batch_x, batch_p)
        epoch_loss.append(current_loss)

    avg_epoch_loss = np.mean(epoch_loss) if epoch_loss else 0.0
    print(f"Epoch {epoch+1}/{epochs} - Average Loss: {avg_epoch_loss:.6f}")

print("End-to-end training finished.")

# --- 7. Save Trained Clustering Model ---
clustering_model.save(trained_model_save_path)
print(f"Trained clustering model saved to {trained_model_save_path}")

# --- 8. Report Final Loss and Cluster Assignments ---
final_q_values = clustering_model.predict(scaled_data, verbose=0) # This is a numpy array
final_assignments = np.argmax(final_q_values, axis=1)

# Convert final_q_values (numpy array) to a TensorFlow tensor for P calculation
final_q_values_tf = tf.convert_to_tensor(final_q_values, dtype=tf.float32)

# Calculate final P for loss reporting, using TensorFlow operations
num_final_p_tf = final_q_values_tf**2 / tf.reduce_sum(final_q_values_tf, axis=0)
final_p_values_tf = num_final_p_tf / tf.reduce_sum(num_final_p_tf, axis=1, keepdims=True)

# KLD loss expects (P, Q) - both should be tensors here
# final_q_values is still numpy, so use final_q_values_tf
final_kld_loss = loss_fn(final_p_values_tf, final_q_values_tf).numpy()

print(f"Final KLD loss against self-derived P: {final_kld_loss:.6f}")
print(f"Final cluster assignments for dummy data (first 20 samples):\n{final_assignments[:20]}")
unique_clusters, counts = np.unique(final_assignments, return_counts=True)
print(f"Cluster distribution: {dict(zip(unique_clusters, counts))}")
