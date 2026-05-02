import re
import pickle
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from config import SEQ_LENGTH, VOCAB_SIZE_LIMIT

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s\']', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def prepare_data(file_path):
    print("Loading and cleaning data...")
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()
    
    text = clean_text(text)
    print(f"Total characters: {len(text):,}")

    # Limit vocabulary for better memory usage
    tokenizer = Tokenizer(num_words=VOCAB_SIZE_LIMIT, oov_token='<OOV>')
    tokenizer.fit_on_texts([text])

    total_words = min(VOCAB_SIZE_LIMIT, len(tokenizer.word_index) + 1)
    print(f"Total words in vocabulary: {total_words}")

    # Create sequences
    print("Creating sequences...")
    tokens = tokenizer.texts_to_sequences([text])[0]
    
    input_sequences = []
    for i in range(SEQ_LENGTH, len(tokens)):
        seq = tokens[i-SEQ_LENGTH:i+1]
        input_sequences.append(seq)

    input_sequences = np.array(pad_sequences(input_sequences, maxlen=SEQ_LENGTH+1, padding='pre'))

    X = input_sequences[:, :-1]
    y = input_sequences[:, -1]

    print(f"X shape: {X.shape} | y shape: {y.shape}")
    
    # Save tokenizer
    with open('tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
    
    return X, y, tokenizer, total_words