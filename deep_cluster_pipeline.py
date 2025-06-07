import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D, Flatten, Reshape, Layer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import KLDivergence, MeanSquaredError
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.utils import shuffle

# --- ClusteringLayer Definition ---
@tf.keras.utils.register_keras_serializable(package='Custom', name='ClusteringLayer')
class ClusteringLayer(Layer):
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

# --- 1. Data Loading and Preprocessing ---
def load_preprocess_data(data_path=None, n_timesteps=24, n_features=1, num_samples_dummy=100):
    print("--- Loading and Preprocessing Data ---")
    if data_path and os.path.exists(data_path):
        print(f"Loading real data from {data_path} - NOT IMPLEMENTED YET, USING DUMMY.")
        # Placeholder for real data loading:
        # import pandas as pd
        # df = pd.read_csv(data_path)
        # # Example: Assuming data is in columns, each row a sample
        # # raw_data = df.values[:, :n_timesteps].astype(np.float32)
        # # # Apply scaling, e.g., per time series (row-wise) to [0,1] or Z-score
        # # if raw_data.ndim == 1: raw_data = np.expand_dims(raw_data, axis=0)
        # # if raw_data.shape[1] != n_timesteps: raise ValueError("Mismatched timesteps")
        # # scaled_data_list = []
        # # for i in range(raw_data.shape[0]):
        # #     series = raw_data[i]
        # #     min_s, max_s = np.min(series), np.max(series)
        # #     scaled_s = (series - min_s) / (max_s - min_s) if (max_s - min_s) > 1e-6 else np.zeros_like(series)
        # #     scaled_data_list.append(scaled_s)
        # # scaled_data = np.array(scaled_data_list)
        # # if n_features == 1: scaled_data = np.expand_dims(scaled_data, axis=-1)
        # # print(f"Loaded and processed real data. Shape: {scaled_data.shape}")
        # # return scaled_data
        pass # Fall through to dummy data

    np.random.seed(42)
    dummy_data = np.random.rand(num_samples_dummy, n_timesteps, n_features).astype(np.float32)
    for i in range(num_samples_dummy):
        peak_time = np.random.randint(2, n_timesteps - 3) if n_timesteps > 5 else n_timesteps // 2
        peak_val = np.random.uniform(0.5, 1.0)
        # Ensure slicing is within bounds for peak_time and array length 5
        start_idx = max(0, peak_time - 2)
        end_idx = min(n_timesteps, peak_time + 3)
        slice_len = end_idx - start_idx
        peak_shape_array = np.array([0.5,0.8,1.0,0.8,0.5])[:slice_len]
        if dummy_data[i, start_idx:end_idx, 0].shape == peak_shape_array.shape:
             dummy_data[i, start_idx:end_idx, 0] = peak_val * peak_shape_array
        dummy_data[i] += 0.1 * np.random.randn(n_timesteps, n_features).astype(np.float32)

    min_val, max_val = np.min(dummy_data), np.max(dummy_data)
    scaled_data = (dummy_data - min_val) / (max_val - min_val) if (max_val - min_val) > 1e-6 else np.zeros_like(dummy_data)
    if n_features == 1 and len(scaled_data.shape) == 2: # Ensure (samples, timesteps, 1)
        scaled_data = np.expand_dims(scaled_data, axis=-1)
    print(f"Using dummy data. Shape: {scaled_data.shape}")
    return scaled_data

# --- 2. Autoencoder Definition ---
def build_autoencoder_and_encoder(input_shape_tuple, embedding_dim):
    print("--- Building Autoencoder ---")
    input_timesteps, input_features = input_shape_tuple
    input_seq = Input(shape=(input_timesteps, input_features), name="ae_input")
    x = Conv1D(16, 3, activation='relu', padding='same')(input_seq)
    x = MaxPooling1D(2, padding='same')(x)
    x = Conv1D(8, 3, activation='relu', padding='same')(x)
    x = MaxPooling1D(2, padding='same')(x)
    shape_before_flatten = tf.keras.backend.int_shape(x)[1:] # (e.g., 6, 8)
    x = Flatten()(x)
    encoded = Dense(embedding_dim, activation='relu', name='embedding')(x)
    encoder = Model(input_seq, encoded, name="encoder")

    latent_inputs = Input(shape=(embedding_dim,), name='decoder_input')
    x = Dense(np.prod(shape_before_flatten), activation='relu')(latent_inputs) # 6*8 = 48
    x = Reshape(shape_before_flatten)(x)
    x = UpSampling1D(2)(x)
    x = Conv1D(8, 3, activation='relu', padding='same')(x)
    x = UpSampling1D(2)(x)
    x = Conv1D(16, 3, activation='relu', padding='same')(x)
    decoded = Conv1D(input_features, 3, activation='sigmoid', padding='same', name="ae_output")(x) # sigmoid for [0,1] data
    decoder = Model(latent_inputs, decoded, name="decoder")

    autoencoder_output = decoder(encoder(input_seq))
    autoencoder = Model(input_seq, autoencoder_output, name="autoencoder")
    return autoencoder, encoder

# --- 3. Pre-training AE ---
def pretrain_ae(autoencoder, data, epochs=50, batch_size=32, verbose=1):
    print("--- Pre-training Autoencoder ---")
    autoencoder.compile(optimizer=Adam(), loss=MeanSquaredError())
    autoencoder.fit(data, data, epochs=epochs, batch_size=batch_size, shuffle=True, verbose=verbose)
    print("Autoencoder pre-training finished.")

# --- 4. Build and Initialize Clustering Model (DEC-style) ---
def build_and_init_clustering_model(encoder_model, n_clusters, data_for_init):
    print("--- Building and Initializing Clustering Model ---")
    embeddings = encoder_model.predict(data_for_init, verbose=0)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    kmeans.fit(embeddings)
    initial_centroids = kmeans.cluster_centers_.astype(np.float32)

    inp = encoder_model.input
    out_emb = encoder_model.output
    clustering_layer_instance = ClusteringLayer(n_clusters=n_clusters, name='clustering_probs')
    clustering_output = clustering_layer_instance(out_emb)
    clustering_model = Model(inputs=inp, outputs=clustering_output, name='clustering_model')
    clustering_layer_instance.set_weights([initial_centroids])
    return clustering_model

# --- 5. Train Clustering Model (DEC training loop) ---
def train_clustering_model(clustering_model, data, epochs=30, batch_size=32, tol=0.001, update_interval_P=1, verbose=1):
    print("--- Training Clustering Model (DEC) ---")
    clustering_model.compile(optimizer=Adam(learning_rate=0.001), loss=KLDivergence())

    y_pred_last = np.zeros(data.shape[0], dtype=int)
    p_all = np.zeros((data.shape[0], clustering_model.output_shape[-1]), dtype=np.float32) # Initialize p_all

    for epoch in range(epochs):
        if epoch % update_interval_P == 0:
            q_all_numpy = clustering_model.predict(data, verbose=0)
            q_all_tf = tf.convert_to_tensor(q_all_numpy, dtype=tf.float32)

            num_p = q_all_tf**2 / tf.reduce_sum(q_all_tf, axis=0)
            p_all_tf = num_p / tf.reduce_sum(num_p, axis=1, keepdims=True)
            p_all = p_all_tf.numpy()

            y_pred_current = np.argmax(q_all_numpy, axis=1)
            delta_label = np.sum(y_pred_current != y_pred_last).astype(float) / y_pred_current.shape[0]
            y_pred_last = y_pred_current

            if verbose: print(f"Epoch {epoch+1}/{epochs} - P updated. Labels changed: {delta_label:.4f}")
            if epoch > 0 and delta_label < tol:
                print(f"Converged at epoch {epoch+1}.")
                break

        shuffled_data, shuffled_p = shuffle(data, p_all, random_state=epoch) # Use current p_all
        epoch_loss_list = []
        for batch_idx in range(int(np.ceil(data.shape[0] / batch_size))):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, data.shape[0])
            if start_idx >= end_idx: continue
            loss = clustering_model.train_on_batch(shuffled_data[start_idx:end_idx], shuffled_p[start_idx:end_idx])
            epoch_loss_list.append(loss)
        if verbose: print(f"Epoch {epoch+1}/{epochs} - Avg KLD Loss: {np.mean(epoch_loss_list) if epoch_loss_list else 0.0:.6f}")
    print("Clustering model training finished.")

# --- 6. Evaluate Clustering ---
def evaluate_clustering_results(encoder_eval_model, dec_eval_model, data):
    print("--- Evaluating Clustering Results ---")
    embeddings = encoder_eval_model.predict(data, verbose=0)
    q_values = dec_eval_model.predict(data, verbose=0)
    hard_assignments = np.argmax(q_values, axis=1)

    if len(np.unique(hard_assignments)) < 2:
        print("Evaluation metrics require at least 2 clusters. Found only 1.")
        return
    if embeddings.shape[0] <= 1: # silhouette_score needs > 1 sample
        print("Not enough samples to calculate Silhouette Score.")
        return

    print(f"Silhouette Score: {silhouette_score(embeddings, hard_assignments):.4f}")
    print(f"Davies-Bouldin Index: {davies_bouldin_score(embeddings, hard_assignments):.4f}")
    print(f"Calinski-Harabasz Index: {calinski_harabasz_score(embeddings, hard_assignments):.4f}")
    unique, counts = np.unique(hard_assignments, return_counts=True)
    print(f"Cluster distribution: {dict(zip(unique, counts))}")

# --- Main Execution Block ---
if __name__ == "__main__":
    print(f"TensorFlow version: {tf.__version__}")

    # --- Hyperparameters & Configuration ---
    N_TIMESTEPS = 24
    N_FEATURES = 1
    EMBEDDING_DIM = 10
    N_CLUSTERS = 3
    NUM_DUMMY_SAMPLES = 100

    AE_PRETRAIN_EPOCHS = 5 # Reduced for quick pipeline test
    AE_BATCH_SIZE = 32
    AE_VERBOSE = 0 # 0 for silent, 1 for progress bar

    DEC_TRAIN_EPOCHS = 5 # Reduced for quick pipeline test
    DEC_BATCH_SIZE = 32
    DEC_TOLERANCE = 0.001
    DEC_UPDATE_INTERVAL_P = 1
    DEC_VERBOSE = 1

    # --- Pipeline Execution ---
    # 1. Load Data
    preprocessed_data = load_preprocess_data(
        data_path=None,
        n_timesteps=N_TIMESTEPS,
        n_features=N_FEATURES,
        num_samples_dummy=NUM_DUMMY_SAMPLES
    )

    # 2. Build and Pre-train Autoencoder
    autoencoder_model, encoder_model_initial = build_autoencoder_and_encoder(
        input_shape_tuple=(N_TIMESTEPS, N_FEATURES),
        embedding_dim=EMBEDDING_DIM
    )
    pretrain_ae(
        autoencoder_model,
        preprocessed_data,
        epochs=AE_PRETRAIN_EPOCHS,
        batch_size=AE_BATCH_SIZE,
        verbose=AE_VERBOSE
    )

    # 3. Build and Train Clustering Model (DEC)
    # Use the encoder_model_initial (weights are from pretraining)
    dec_model = build_and_init_clustering_model(
        encoder_model=encoder_model_initial,
        n_clusters=N_CLUSTERS,
        data_for_init=preprocessed_data
    )
    train_clustering_model(
        clustering_model=dec_model,
        data=preprocessed_data,
        epochs=DEC_TRAIN_EPOCHS,
        batch_size=DEC_BATCH_SIZE,
        tol=DEC_TOLERANCE,
        update_interval_P=DEC_UPDATE_INTERVAL_P,
        verbose=DEC_VERBOSE
    )

    # 4. Evaluate
    # For evaluation, use the encoder_model_initial whose weights were pre-trained
    # and potentially fine-tuned during DEC model training (if its layers were trainable).
    # The encoder_model_initial is the first "layer" (functional model) of dec_model.
    # Its weights are updated when dec_model is trained.
    evaluate_clustering_results(encoder_model_initial, dec_model, preprocessed_data)

    print("\n--- Deep Clustering Pipeline Finished ---")
