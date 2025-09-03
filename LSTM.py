import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
# Updated imports to include the new layers
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# --- 1. Load and Preprocess the IMDB Dataset ---

max_features = 10000  # Vocabulary size
max_len = 500         # Max length of reviews

print("Loading and preprocessing data...")
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_features)

# Pad sequences to a uniform length
X_train = sequence.pad_sequences(X_train, maxlen=max_len)
X_test = sequence.pad_sequences(X_test, maxlen=max_len)
print("Data preprocessing complete.")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")


# --- 2. Build the Improved Bidirectional LSTM Model ---

print("\nBuilding the improved model with Bidirectional LSTM and Dropout...")
model = Sequential()

# Embedding Layer
model.add(Embedding(input_dim=max_features, output_dim=128, input_length=max_len))
model.add(Dropout(0.5))

# Bidirectional LSTM Layer (Replaces SimpleRNN)
model.add(Bidirectional(LSTM(64)))
model.add(Dropout(0.5))

# Output Layer
model.add(Dense(1, activation='sigmoid'))

# --- 3. Compile the Model ---

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()


# --- 4. Train the Model ---

print("\nStarting model training...")
# Create an instance of EarlyStopping Callback
earlystopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2,
    callbacks=[earlystopping]
)
print("Model training complete.")


# --- 5. Save the Trained Model ---

# Save the new, more powerful model with a descriptive name
model.save('bilstm_imdb_model.h5')
print("\nModel saved successfully as 'bilstm_imdb_model.h5'")
