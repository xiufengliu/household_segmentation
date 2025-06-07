import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D, Flatten, Reshape, Layer, Attention, Concatenate
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
def load_preprocess_data(load_shape_path=None, weather_data_path=None,
                         n_timesteps=24, n_load_features=1, n_weather_features=2,
                         num_samples_dummy=100):
    print("--- Loading and Preprocessing Data (Load Shapes & Weather) ---")

    # Placeholder for real data loading
    if load_shape_path and weather_data_path and os.path.exists(load_shape_path) and os.path.exists(weather_data_path):
        print(f"Attempting to load real data from {load_shape_path} and {weather_data_path}")
        # TODO: Implement real data loading for load shapes
        # import pandas as pd
        # df_load = pd.read_csv(load_shape_path)
        # ... extract relevant columns, handle timestamps, align ...
        # raw_load_shapes = df_load.values

        # TODO: Implement real data loading for weather
        # df_weather = pd.read_csv(weather_data_path)
        # ... extract relevant columns, handle timestamps, align with load shapes ...
        # raw_weather_data = df_weather.values

        # Ensure alignment and consistent number of samples and timesteps
        # This part is crucial and depends on data format.

        # Normalization for real data:
        # scaled_load_shapes = ... (e.g., Z-score per series, or MinMax per series)
        # scaled_weather_data = ... (e.g., Z-score for temp, MinMax for humidity)
        print("Real data loading not implemented yet. Falling back to dummy data.")
        # For now, fall through to dummy data generation
        pass # Fall through to dummy data generation

    # Generate Dummy Data if real data not loaded
    np.random.seed(42)

    # Dummy Load Shapes
    dummy_load_shapes = np.random.rand(num_samples_dummy, n_timesteps, n_load_features).astype(np.float32)
    for i in range(num_samples_dummy): # Add some structure
        peak_time = np.random.randint(2, n_timesteps - 3) if n_timesteps > 5 else n_timesteps // 2
        peak_val = np.random.uniform(0.5, 1.0)
        start_idx = max(0, peak_time - 2)
        end_idx = min(n_timesteps, peak_time + 3)
        slice_len = end_idx - start_idx
        peak_shape_array = np.array([0.5,0.8,1.0,0.8,0.5])[:slice_len]

        if dummy_load_shapes[i, start_idx:end_idx, 0].shape == peak_shape_array.shape:
             dummy_load_shapes[i, start_idx:end_idx, 0] = peak_val * peak_shape_array
        dummy_load_shapes[i] += 0.1 * np.random.randn(n_timesteps, n_load_features).astype(np.float32)

    # Dummy Weather Data (Temperature & Humidity)
    dummy_weather_data = np.zeros((num_samples_dummy, n_timesteps, n_weather_features), dtype=np.float32)
    time_axis = np.linspace(0, 2*np.pi, n_timesteps) # Represents 24 hours
    for i in range(num_samples_dummy):
        # Temperature: Sinusoidal daily pattern + noise
        temp_amplitude = np.random.uniform(5, 15)
        temp_offset = np.random.uniform(5, 15) # Base temperature
        temp_daily_cycle = temp_offset + temp_amplitude * np.sin(time_axis + np.random.uniform(0, np.pi))
        dummy_weather_data[i, :, 0] = temp_daily_cycle + np.random.normal(0, 2, n_timesteps) # Add some noise

        # Humidity: Inverse sinusoidal pattern (simplistic) + noise
        hum_amplitude = np.random.uniform(10, 30)
        hum_offset = np.random.uniform(40, 60) # Base humidity
        hum_daily_cycle = hum_offset - hum_amplitude * np.sin(time_axis + np.random.uniform(0, np.pi))
        dummy_weather_data[i, :, 1] = hum_daily_cycle + np.random.normal(0, 5, n_timesteps)
        dummy_weather_data[i, :, 1] = np.clip(dummy_weather_data[i, :, 1], 0, 100) # Clip humidity to [0,100]

    print(f"Generated dummy load shapes. Shape: {dummy_load_shapes.shape}")
    print(f"Generated dummy weather data. Shape: {dummy_weather_data.shape}")

    # Normalization for Dummy Data
    # Load Shapes: global min-max to [0,1]
    ls_min, ls_max = np.min(dummy_load_shapes), np.max(dummy_load_shapes)
    scaled_load_shapes = (dummy_load_shapes - ls_min) / (ls_max - ls_min) if (ls_max - ls_min) > 1e-6 else np.zeros_like(dummy_load_shapes)

    # Weather Data: global min-max to [0,1] for each feature for simplicity
    scaled_weather_data = np.zeros_like(dummy_weather_data)
    for j in range(n_weather_features):
        feat_min, feat_max = np.min(dummy_weather_data[:,:,j]), np.max(dummy_weather_data[:,:,j])
        if (feat_max - feat_min) > 1e-6:
            scaled_weather_data[:,:,j] = (dummy_weather_data[:,:,j] - feat_min) / (feat_max - feat_min)
        else:
            scaled_weather_data[:,:,j] = np.zeros_like(dummy_weather_data[:,:,j])

    if n_load_features == 1 and len(scaled_load_shapes.shape) == 2: # Ensure (samples, timesteps, 1)
        scaled_load_shapes = np.expand_dims(scaled_load_shapes, axis=-1)

    print(f"Scaled load shapes shape: {scaled_load_shapes.shape}")
    print(f"Scaled weather data shape: {scaled_weather_data.shape}")

    return scaled_load_shapes, scaled_weather_data

# --- 2. Weather Fused Encoder Definition ---
def build_weather_fused_encoder(load_shape_input_shape_tuple, weather_input_shape_tuple, embedding_dim):
    print("--- Building Weather Fused Encoder with Attention ---")
    ls_timesteps, ls_features = load_shape_input_shape_tuple
    weather_timesteps, weather_features = weather_input_shape_tuple

    # Inputs
    load_shape_input = Input(shape=(ls_timesteps, ls_features), name='load_shape_input')
    weather_input = Input(shape=(weather_timesteps, weather_features), name='weather_input')

    # Load Shape Stream (example)
    ls_x = Conv1D(16, 3, activation='relu', padding='same', name='ls_conv1')(load_shape_input)
    ls_x = MaxPooling1D(2, padding='same', name='ls_pool1')(ls_x) # 24 -> 12
    ls_intermediate_repr = Conv1D(8, 3, activation='relu', padding='same', name='ls_conv2')(ls_x)
    ls_intermediate_repr = MaxPooling1D(2, padding='same', name='ls_pool2')(ls_intermediate_repr) # 12 -> 6. Shape: (None, 6, 8)

    # Weather Stream (example)
    w_x = Conv1D(8, 3, activation='relu', padding='same', name='w_conv1')(weather_input)
    w_x = MaxPooling1D(2, padding='same', name='w_pool1')(w_x) # 24 -> 12
    w_intermediate_repr = Conv1D(4, 3, activation='relu', padding='same', name='w_conv2')(w_x)
    w_intermediate_repr = MaxPooling1D(2, padding='same', name='w_pool2')(w_intermediate_repr) # 12 -> 6. Shape: (None, 6, 4)

    print(f"Load shape intermediate output shape: {ls_intermediate_repr.shape}")
    print(f"Weather intermediate output shape: {w_intermediate_repr.shape}")

    # Attention Fusion: Load shape queries weather
    # Project weather stream's features to match load_shape stream's features for QK compatibility
    w_intermediate_repr_projected = Conv1D(filters=ls_intermediate_repr.shape[-1], kernel_size=1, activation='relu', padding='same', name='w_feat_projection')(w_intermediate_repr)
    print(f"Weather intermediate projected shape: {w_intermediate_repr_projected.shape}") # Expected: (None, 6, 8)

    attention_layer = Attention(use_scale=True, name='weather_attention')
    weather_context = attention_layer([ls_intermediate_repr, w_intermediate_repr_projected])
    print(f"Weather context (attention output) shape: {weather_context.shape}") # Expected: (None, 6, 8)

    # Combine Representations
    fused_sequence = Concatenate(axis=-1, name='concatenate_ls_weather')([ls_intermediate_repr, weather_context]) # Shape (None, 6, 16)
    print(f"Fused sequence shape: {fused_sequence.shape}")

    # Final Embedding
    flattened_fused = Flatten(name='flatten_fused')(fused_sequence)
    fused_embedding = Dense(embedding_dim, activation='relu', name='fused_embedding')(flattened_fused)
    print(f"Final fused embedding shape: {fused_embedding.shape}")

    # Define the Encoder Model
    encoder_model = Model(inputs=[load_shape_input, weather_input], outputs=fused_embedding, name='weather_fused_encoder')

    return encoder_model

# --- 3. Weather Fused Autoencoder and Pre-training ---
def build_weather_fused_autoencoder(load_shape_input_shape_tuple, weather_input_shape_tuple, embedding_dim):
    print("--- Building Weather Fused Autoencoder ---")
    ls_timesteps, ls_features = load_shape_input_shape_tuple

    fused_encoder_model = build_weather_fused_encoder(
        load_shape_input_shape_tuple,
        weather_input_shape_tuple,
        embedding_dim
    )
    fused_embedding = fused_encoder_model.output

    # Decoder for load shape reconstruction from fused_embedding
    decoder_input_shape_prod = (ls_timesteps // 4) * 8

    x = Dense(decoder_input_shape_prod, activation='relu', name='dec_dense')(fused_embedding)
    x = Reshape((ls_timesteps // 4, 8), name='dec_reshape')(x)

    x = UpSampling1D(2, name='dec_upsample1')(x)
    x = Conv1D(8, 3, activation='relu', padding='same', name='dec_conv1')(x)
    x = UpSampling1D(2, name='dec_upsample2')(x)
    reconstructed_load_shape = Conv1D(ls_features, 3, activation='sigmoid', padding='same', name='dec_output_ls')(x)

    autoencoder_model = Model(
        inputs=fused_encoder_model.inputs,
        outputs=reconstructed_load_shape,
        name='weather_fused_autoencoder'
    )

    return autoencoder_model, fused_encoder_model

def pretrain_weather_ae(autoencoder_model, scaled_load_shapes, scaled_weather_data, epochs=50, batch_size=32, verbose=1):
    print("--- Pre-training Weather Fused Autoencoder ---")
    autoencoder_model.compile(optimizer=Adam(), loss=MeanSquaredError())
    history = autoencoder_model.fit(
        [scaled_load_shapes, scaled_weather_data],
        scaled_load_shapes,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        verbose=verbose
    )
    print("Weather Fused Autoencoder pre-training finished.")
    return history

# --- 4. Build and Initialize Clustering Model (DEC-style) ---
def build_and_init_dec_model_with_fused_encoder(
        fused_encoder_model,
        n_clusters,
        load_shapes_for_init,
        weather_data_for_init):
    print("--- Building and Initializing DEC Model with Fused Encoder ---")

    print("Generating embeddings for K-Means initialization...")
    embeddings = fused_encoder_model.predict(
        [load_shapes_for_init, weather_data_for_init],
        verbose=0
    )
    print(f"Embeddings shape for K-Means: {embeddings.shape}")

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    kmeans.fit(embeddings)
    initial_centroids = kmeans.cluster_centers_.astype(np.float32)
    print(f"Initial K-Means centroids shape: {initial_centroids.shape}")

    dec_model_inputs = fused_encoder_model.inputs
    fused_embeddings_output = fused_encoder_model.output

    clustering_layer_instance = ClusteringLayer(n_clusters=n_clusters, name='clustering_probs')
    clustering_output = clustering_layer_instance(fused_embeddings_output)

    dec_model = Model(inputs=dec_model_inputs, outputs=clustering_output, name='dec_model_with_fused_encoder')

    clustering_layer_instance.set_weights([initial_centroids])
    print("Set initial weights of ClusteringLayer with K-Means centroids.")

    return dec_model

# --- 5. Train Clustering Model (DEC training loop) ---
def train_clustering_model(
        clustering_model,
        list_of_data_inputs,
        epochs=30,
        batch_size=32,
        tol=0.001,
        update_interval_P=1,
        verbose=1):

    print("--- Training Clustering Model (DEC) with Fused Encoder ---")

    print("Compiling clustering model for DEC training (Adam, KLD loss)")
    clustering_model.compile(optimizer=Adam(learning_rate=0.001), loss=KLDivergence())

    num_samples = list_of_data_inputs[0].shape[0]
    y_pred_last = np.zeros(num_samples, dtype=int)
    q_init = clustering_model.predict(list_of_data_inputs, verbose=0)
    q_init_tf = tf.convert_to_tensor(q_init, dtype=tf.float32)
    num_p_init = q_init_tf**2 / tf.reduce_sum(q_init_tf, axis=0)
    p_all_tf_init = num_p_init / tf.reduce_sum(num_p_init, axis=1, keepdims=True)
    p_all = p_all_tf_init.numpy()

    for epoch in range(epochs):
        epoch_loss_list = []

        if epoch % update_interval_P == 0:
            if verbose: print(f"Updating target distribution P for epoch {epoch+1}...")
            q_all_numpy = clustering_model.predict(list_of_data_inputs, verbose=0)
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

        shuffled_inputs_and_p = shuffle(*list_of_data_inputs, p_all, random_state=epoch)

        shuffled_p = shuffled_inputs_and_p[-1]
        shuffled_data_inputs = list(shuffled_inputs_and_p[:-1])

        num_batches = int(np.ceil(num_samples / batch_size))
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_samples)
            if start_idx >= end_idx: continue

            current_batch_x_list = [data_arr[start_idx:end_idx] for data_arr in shuffled_data_inputs]
            current_batch_p = shuffled_p[start_idx:end_idx]

            loss = clustering_model.train_on_batch(current_batch_x_list, current_batch_p)
            epoch_loss_list.append(loss)

        avg_epoch_loss = np.mean(epoch_loss_list) if epoch_loss_list else 0.0
        if verbose: print(f"Epoch {epoch+1}/{epochs} - Avg KLD Loss: {avg_epoch_loss:.6f}")

    print("Clustering model training finished.")
    final_q_values = clustering_model.predict(list_of_data_inputs, verbose=0)
    return final_q_values

# --- 6. Evaluate Clustering ---
def evaluate_clustering_results(
        encoder_model_to_get_embeddings,
        dec_model_to_get_assignments,
        list_of_data_inputs):
    print("--- Evaluating Clustering Results (Weather Fusion Model) ---")

    if not isinstance(list_of_data_inputs, list):
        print("Warning: list_of_data_inputs was not a list in evaluate_clustering_results. Wrapping.")
        list_of_data_inputs = [list_of_data_inputs]

    print("Generating final embeddings...")
    embeddings = encoder_model_to_get_embeddings.predict(list_of_data_inputs, verbose=0)
    print(f"Final embeddings shape: {embeddings.shape}")

    print("Generating final cluster assignments...")
    q_values = dec_model_to_get_assignments.predict(list_of_data_inputs, verbose=0)
    hard_assignments = np.argmax(q_values, axis=1)
    print(f"Final assignments shape: {hard_assignments.shape}")

    if len(np.unique(hard_assignments)) < 2:
        print("Evaluation metrics (Silhouette, Davies-Bouldin, Calinski-Harabasz) require at least 2 clusters.")
        unique_clusters, counts = np.unique(hard_assignments, return_counts=True)
        print(f"Cluster distribution: {dict(zip(unique_clusters, counts))}")
        return
    if embeddings.shape[0] <= 1:
        print("Not enough samples to calculate Silhouette Score for evaluation.")
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
    N_LOAD_FEATURES = 1
    N_WEATHER_FEATURES = 2
    EMBEDDING_DIM = 10
    N_CLUSTERS = 3
    NUM_DUMMY_SAMPLES = 100

    AE_PRETRAIN_EPOCHS = 5
    AE_BATCH_SIZE = 32
    AE_VERBOSE = 0

    DEC_TRAIN_EPOCHS = 5
    DEC_BATCH_SIZE = 32
    DEC_TOLERANCE = 0.001
    DEC_UPDATE_INTERVAL_P = 1
    DEC_VERBOSE = 1

    # --- Pipeline Execution ---
    # 1. Load Data
    scaled_load_shapes, scaled_weather_data = load_preprocess_data(
        load_shape_path=None,
        weather_data_path=None,
        n_timesteps=N_TIMESTEPS,
        n_load_features=N_LOAD_FEATURES,
        n_weather_features=N_WEATHER_FEATURES,
        num_samples_dummy=NUM_DUMMY_SAMPLES
    )

    print(f"Main: Loaded scaled_load_shapes with shape: {scaled_load_shapes.shape}")
    print(f"Main: Loaded scaled_weather_data with shape: {scaled_weather_data.shape}")

    # 2. Build Weather Fused Autoencoder (which includes the fused encoder)
    weather_ae_model, weather_encoder_model = build_weather_fused_autoencoder(
        load_shape_input_shape_tuple=(N_TIMESTEPS, N_LOAD_FEATURES),
        weather_input_shape_tuple=(N_TIMESTEPS, N_WEATHER_FEATURES),
        embedding_dim=EMBEDDING_DIM
    )
    print("\n--- Weather Fused Autoencoder Summary ---")
    weather_ae_model.summary(line_length=120)

    # 3. Pre-train the Weather Fused Autoencoder
    pretrain_history = pretrain_weather_ae(
        weather_ae_model,
        scaled_load_shapes,
        scaled_weather_data,
        epochs=AE_PRETRAIN_EPOCHS,
        batch_size=AE_BATCH_SIZE,
        verbose=AE_VERBOSE
    )
    if pretrain_history and pretrain_history.history.get('loss'):
        print(f"Final pre-training loss: {pretrain_history.history['loss'][-1]:.4f}")
    else:
        print("Pre-training did not return history or loss.")

    print("\nBuilding DEC model with the pre-trained fused encoder...")
    dec_model_wf = build_and_init_dec_model_with_fused_encoder(
        fused_encoder_model=weather_encoder_model,
        n_clusters=N_CLUSTERS,
        load_shapes_for_init=scaled_load_shapes,
        weather_data_for_init=scaled_weather_data
    )
    print("\n--- DEC Model with Fused Encoder Summary ---")
    dec_model_wf.summary(line_length=120)

    # 4. Train DEC Model
    print("\nStarting End-to-End Training of DEC Model with Weather Fusion...")
    final_q_values_wf = train_clustering_model(
        clustering_model=dec_model_wf,
        list_of_data_inputs=[scaled_load_shapes, scaled_weather_data],
        epochs=DEC_TRAIN_EPOCHS,
        batch_size=DEC_BATCH_SIZE,
        tol=DEC_TOLERANCE,
        update_interval_P=DEC_UPDATE_INTERVAL_P,
        verbose=DEC_VERBOSE
    )

    final_assignments_wf = np.argmax(final_q_values_wf, axis=1)
    unique_clusters_wf, counts_wf = np.unique(final_assignments_wf, return_counts=True)
    print(f"Final cluster assignments (weather fusion): {dict(zip(unique_clusters_wf, counts_wf))}")

    # 5. Evaluate DEC Model
    print("\nStarting Evaluation of DEC Model with Weather Fusion...")

    if not dec_model_wf.layers:
        print("ERROR: dec_model_wf has no layers. Cannot extract encoder for evaluation.")
    else:
        encoder_output_tensor_final = dec_model_wf.layers[-2].output
        final_trained_fused_encoder = Model(
            inputs=dec_model_wf.inputs,
            outputs=encoder_output_tensor_final,
            name="final_trained_fused_encoder"
        )
        print("\n--- Final Trained Fused Encoder (for evaluation) Summary ---")
        final_trained_fused_encoder.summary(line_length=120)

        evaluate_clustering_results(
            encoder_model_to_get_embeddings=final_trained_fused_encoder,
            dec_model_to_get_assignments=dec_model_wf,
            list_of_data_inputs=[scaled_load_shapes, scaled_weather_data]
        )

    # --- Plan for Weather Impact Analysis (Printed Output) ---
    print("\n--- Plan for Weather Impact Analysis (with Real Data) ---")
    print("1. **Comparative Evaluation:**")
    print("   - Compare Silhouette, Davies-Bouldin, Calinski-Harabasz scores against the non-weather DEC model.")
    print("   - If external labels or business KPIs exist, use ARI, NMI, or task-specific metrics.")
    print("2. **Attention Mechanism Analysis:**")
    print("   - The 'weather_fused_encoder' contains an 'Attention' layer (e.g., named 'weather_attention').")
    print("   - To visualize attention: Modify the 'weather_fused_encoder' to output attention scores in addition to the embedding.")
    print("     Alternatively, build a new model with the same weights that outputs attention scores.")
    print("     `attention_layer = weather_fused_encoder.get_layer('weather_attention')`")
    print("     `attention_scores_model = Model(inputs=weather_fused_encoder.inputs, outputs=attention_layer.output[1]) # If it's Bahdanau, output[1] is often scores`")
    print("     Or, if it's a simple Attention layer, it might involve inspecting its intermediate calculations if not directly outputted.")
    print("   - For selected samples (e.g., cluster exemplars), predict attention scores.")
    print("   - Plot heatmaps of attention: load shape time steps vs. weather time steps (or weather features over time).")
    print("   - Analyze: Does the model attend to high/low temperatures during peak load times? Does it ignore weather when load is baseload?")
    print("3. **Cluster Characterization vs. Weather:**")
    print("   - For each cluster, analyze the average and distribution of weather features (temp, humidity) for days belonging to that cluster.")
    print("   - Example: 'Cluster A is associated with high afternoon temperatures', 'Cluster B load shapes are less sensitive to humidity changes'.")
    print("   - Visualize average load shapes per cluster alongside average weather patterns for those clusters.")
    print("4. **Qualitative Load Shape Analysis:**")
    print("   - Compare the interpretability and distinctiveness of average cluster load shapes from the weather-fused model versus the non-weather model.")
    print("   - Do the weather-fused clusters represent more coherent daily energy usage patterns once weather influence is accounted for by the model?")
    print("5. **Robustness (Optional):**")
    print("   - Test model performance on subsets of data with extreme weather conditions vs. mild weather conditions.")

    print("\n--- Evaluation and Analysis Planning (Step 7) Complete ---")
