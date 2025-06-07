import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import os

print(f"TensorFlow version: {tf.__version__}")

# --- Parameters ---
num_samples = 100
timesteps = 24
features = 1
epochs = 50
batch_size = 8
model_path = 'autoencoder_model_definition.keras'
weights_path = 'pretrained_autoencoder.weights.h5'

# --- 2. Generate Dummy Data ---
# Using random data for now. When real data is available, this will be replaced by loading and preprocessing it.
np.random.seed(42) # for reproducibility
dummy_data = np.random.rand(num_samples, timesteps, features).astype(np.float32)
# Add some structure to the dummy data to make it a bit more interesting than pure noise
for i in range(num_samples):
    peak_time = np.random.randint(5, 19)
    peak_val = np.random.uniform(0.5, 1.0)
    dummy_data[i, peak_time-2:peak_time+3, 0] = peak_val * np.array([0.5, 0.8, 1.0, 0.8, 0.5])
    dummy_data[i] += 0.1 * np.random.randn(timesteps, features).astype(np.float32) # add some noise


print(f"Generated dummy data with shape: {dummy_data.shape}")

# --- 3. Scale Data to [0,1] ---
# Global min-max scaling for simplicity with dummy data
min_val = np.min(dummy_data)
max_val = np.max(dummy_data)

if max_val == min_val:
    scaled_data = np.zeros_like(dummy_data)
    print("Data is constant, scaled to zeros.")
else:
    scaled_data = (dummy_data - min_val) / (max_val - min_val)
    print(f"Data scaled to [0,1] range. Original min: {min_val:.4f}, max: {max_val:.4f}")

# Reshape if features = 1, to ensure it's (samples, timesteps, 1)
if features == 1 and len(scaled_data.shape) == 2:
    scaled_data = np.expand_dims(scaled_data, axis=-1)
print(f"Scaled data shape: {scaled_data.shape}")


# --- 4. Load Autoencoder Model ---
if not os.path.exists(model_path):
    print(f"ERROR: Autoencoder model definition file not found at {model_path}")
    # In a real workflow, might raise an exception or exit
    # For this subtask, if the model isn't there, we can't proceed.
    # Attempt to create a dummy model if not found, to allow script to run for testing purposes
    # This part should ideally not be needed if previous steps are correct
    print(f"Attempting to create a dummy model as {model_path} was not found.")
    input_seq = tf.keras.layers.Input(shape=(timesteps, features))
    x = tf.keras.layers.Conv1D(16, 3, activation='relu', padding='same')(input_seq)
    x = tf.keras.layers.MaxPooling1D(2, padding='same')(x)
    x = tf.keras.layers.Conv1D(8, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling1D(2, padding='same')(x)
    shape_before_flatten = tf.keras.backend.int_shape(x)[1:]
    x = tf.keras.layers.Flatten()(x)
    encoded = tf.keras.layers.Dense(10, activation='relu', name='embedding')(x)
    # Decoder
    latent_inputs = tf.keras.layers.Input(shape=(10,), name='decoder_input')
    x = tf.keras.layers.Dense(np.prod(shape_before_flatten), activation='relu')(latent_inputs)
    x = tf.keras.layers.Reshape(shape_before_flatten)(x)
    x = tf.keras.layers.UpSampling1D(2)(x)
    x = tf.keras.layers.Conv1D(8, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.UpSampling1D(2)(x)
    x = tf.keras.layers.Conv1D(16, 3, activation='relu', padding='same')(x)
    decoded = tf.keras.layers.Conv1D(features, 3, activation='sigmoid', padding='same')(x)
    # Encoder model
    encoder_model = tf.keras.models.Model(input_seq, encoded)
    # Decoder model
    decoder_model = tf.keras.models.Model(latent_inputs, decoded)
    # Autoencoder model
    autoencoder_output = decoder_model(encoder_model(input_seq))
    autoencoder = tf.keras.models.Model(input_seq, autoencoder_output, name="autoencoder_dummy")
    autoencoder.save(model_path)
    print(f"Dummy model saved to {model_path}. This indicates an issue in the overall workflow.")


print(f"Loading autoencoder model from {model_path}...")
autoencoder = load_model(model_path)
print("Autoencoder model loaded successfully.")
autoencoder.summary() # Verify

# --- 5. Compile Autoencoder ---
autoencoder.compile(optimizer='adam', loss='mse')
print("Autoencoder compiled with Adam optimizer and MSE loss.")

# --- 6. Train Autoencoder ---
print("Starting autoencoder pre-training...")
history = autoencoder.fit(
    scaled_data,  # Input data
    scaled_data,  # Target data (reconstruction)
    epochs=epochs,
    batch_size=batch_size,
    shuffle=True,
    verbose=1 # Set to 1 or 2 to see progress, 0 for silent
)
print("Autoencoder pre-training finished.")

final_loss = history.history['loss'][-1]
print(f"Final training loss (MSE): {final_loss:.6f}")

# --- 7. Save Trained Weights ---
autoencoder.save_weights(weights_path)
print(f"Pre-trained autoencoder weights saved to {weights_path}")

# Additionally, save the full model (including weights) for easier loading later if needed
full_model_save_path = 'pretrained_autoencoder_full_model.keras'
autoencoder.save(full_model_save_path)
print(f"Full pre-trained autoencoder model saved to {full_model_save_path}")
