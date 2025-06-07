import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D, Flatten, Reshape
from tensorflow.keras.models import Model
import numpy as np

print(f"TensorFlow version: {tf.__version__}")

# --- Parameters ---
input_timesteps = 24
input_features = 1 # Univariate time series
embedding_dim = 10

# --- 1. Define Input Layer ---
input_seq = Input(shape=(input_timesteps, input_features))

# --- 2. Encoder Network ---
# Conv1D expects input shape: (batch_size, steps, features)
x = Conv1D(16, 3, activation='relu', padding='same')(input_seq)
x = MaxPooling1D(2, padding='same')(x) # Output: (None, 12, 16)
x = Conv1D(8, 3, activation='relu', padding='same')(x)
x = MaxPooling1D(2, padding='same')(x) # Output: (None, 6, 8)
shape_before_flatten = tf.keras.backend.int_shape(x)[1:] # (6, 8)
x = Flatten()(x)
encoded = Dense(embedding_dim, activation='relu', name='embedding')(x)

encoder = Model(input_seq, encoded, name="encoder")
print("--- Encoder Summary ---")
encoder.summary()

# --- 3. Decoder Network ---
latent_inputs = Input(shape=(embedding_dim,), name='decoder_input')
# Upscale to the shape before flattening in the encoder
x = Dense(np.prod(shape_before_flatten), activation='relu')(latent_inputs) # np.prod((6,8)) = 48
x = Reshape(shape_before_flatten)(x) # Reshape to (6, 8)

x = UpSampling1D(2)(x) # Output: (None, 12, 8)
x = Conv1D(8, 3, activation='relu', padding='same')(x)
x = UpSampling1D(2)(x) # Output: (None, 24, 8)
x = Conv1D(16, 3, activation='relu', padding='same')(x)
# Output layer to reconstruct the original shape (24,1)
# Using 'sigmoid' activation assuming input data will be scaled to [0,1]
# Using kernel_size=3 and 'same' padding to maintain dimension 24
decoded = Conv1D(input_features, 3, activation='sigmoid', padding='same')(x)

decoder = Model(latent_inputs, decoded, name="decoder")
print("\n--- Decoder Summary ---")
decoder.summary()

# --- 4. Create Autoencoder Model ---
autoencoder_output = decoder(encoder(input_seq))
autoencoder = Model(input_seq, autoencoder_output, name="autoencoder")

# --- 5. Print Model Summary ---
print("\n--- Autoencoder Summary ---")
autoencoder.summary()

# Store the model definition (optional, for verification)
try:
    autoencoder.save("autoencoder_model_definition.keras")
    print("\nAutoencoder model definition saved to autoencoder_model_definition.keras")
except Exception as e:
    print(f"\nError saving model definition: {e}")
