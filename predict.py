import pickle
import numpy as np
from tensorflow.keras.models import load_model
from config import SEQ_LENGTH

# Load model and tokenizer
model = load_model('models/predictive_keyboard_lstm.h5')

with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

def predict_next_words(seed_text, num_suggestions=5):
    print(f"\nInput: '{seed_text}'")
    print("Top Predictions:")
    
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=SEQ_LENGTH, padding='pre')
    
    predicted_probs = model.predict(token_list, verbose=0)[0]
    top_indices = np.argsort(predicted_probs)[-num_suggestions:][::-1]
    
    suggestions = []
    for idx in top_indices:
        word = tokenizer.index_word.get(idx, '<OOV>')
        prob = predicted_probs[idx] * 100
        print(f"  → {word:<15} ({prob:.2f}%)")
        suggestions.append(word)
    
    return suggestions

# Test the keyboard
if __name__ == "__main__":
    test_sentences = [
        "bhai lets go to the",
        "i am feeling very",
        "kal office mein",
        "yaar tu kya",
        "today i will"
    ]
    
    for sentence in test_sentences:
        predict_next_words(sentence)