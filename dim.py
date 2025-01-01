from ezr import *



d        = DATA().adds(csv(the.train))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

# Function to create an autoencoder model
def create_autoencoder(input_dim, encoding_dim):
    # Input layer
    input_layer = Input(shape=(input_dim,))

    # Encoding layer
    encoded = Dense(encoding_dim, activation='relu')(input_layer)

    # Decoding layer
    decoded = Dense(input_dim, activation='sigmoid')(encoded)

    # Autoencoder model
    autoencoder = Model(inputs=input_layer, outputs=decoded)

    # Encoder model (for dimensionality reduction)
    encoder = Model(inputs=input_layer, outputs=encoded)

    return autoencoder, encoder



# Preprocess the data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(d.rows)

# Split data into training and testing sets
X_train, X_test = train_test_split(data_scaled, test_size=0.2, random_state=42)

# Hyperparameter: Number of output columns (dimensionality of the encoding layer)
n = 2  # Adjust this value as needed

# Create the autoencoder and encoder
input_dim = X_train.shape[1]
autoencoder, encoder = create_autoencoder(input_dim, n)

# Compile the autoencoder
autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# Train the autoencoder
history = autoencoder.fit(
    X_train, X_train,
    epochs=50,
    batch_size=16,
    shuffle=True,
    validation_data=(X_test, X_test),
    verbose=1
)

# Use the encoder to transform the data
X_train_reduced = encoder.predict(X_train)
X_test_reduced = encoder.predict(X_test)

# Convert back to a DataFrame for better interpretability
columns = [f"encoded_feature_{i+1}" for i in range(n)]
X_train_reduced_df = pd.DataFrame(X_train_reduced, columns=columns)
X_test_reduced_df = pd.DataFrame(X_test_reduced, columns=columns)

# Output the transformed data
print("Reduced Training Data:")
print(X_train_reduced_df.head())
print("\nReduced Testing Data:")
print(X_test_reduced_df.head())


