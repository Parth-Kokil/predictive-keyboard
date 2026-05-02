import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from preprocess import prepare_data
from config import SEQ_LENGTH, EMBEDDING_DIM, LSTM_UNITS, BATCH_SIZE, EPOCHS
import os

# Enable GPU memory growth (important for local)
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("✅ GPU detected and configured")
else:
    print("⚠️ No GPU found. Training will be slow on CPU.")

# Load data
X, y, tokenizer, total_words = prepare_data('data/indian_data.txt')   # Change path as needed

# Build Model
model = Sequential([
    Embedding(total_words, EMBEDDING_DIM, input_length=SEQ_LENGTH),
    LSTM(LSTM_UNITS, return_sequences=True),
    Dropout(0.2),
    LSTM(LSTM_UNITS),
    Dropout(0.2),
    Dense(256, activation='relu'),
    Dense(total_words, activation='softmax')
])

model.compile(
    loss='sparse_categorical_crossentropy',   # Better than one-hot for large vocab
    optimizer='adam',
    metrics=['accuracy']
)

model.summary()

# Callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint('models/best_model.h5', save_best_only=True)
]

print("Starting training...")
history = model.fit(
    X, y,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.1,
    callbacks=callbacks,
    verbose=1
)

# Save final model
model.save('models/predictive_keyboard_lstm.h5')
print("✅ Training completed and model saved!")