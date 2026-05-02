# 🎹 Predictive Keyboard Model

An intelligent **Next-Word Prediction** system trained on Indian English and Hinglish data using LSTM. It provides real-time typing suggestions just like modern smartphone keyboards (Gboard / SwiftKey).

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Gradio](https://img.shields.io/badge/Gradio-FF6F00?style=for-the-badge&logo=gradio&logoColor=white)

##  Features

- Real-time word suggestions as you type
- Trained on Indian English + Hinglish conversational data
- Click-to-insert functionality (like real keyboards)
- Beautiful and responsive web UI
- Lightweight and fast inference

## 🛠️ Tech Stack

- **Language**: Python
- **Model**: LSTM (Long Short-Term Memory)
- **Deep Learning**: TensorFlow / Keras
- **Frontend**: Gradio
- **Data**: Indian English + Hinglish corpus

## 🚀 How to Run Locally

# 1. Clone the repository
git clone https://github.com/Parth-Kokil/predictive-keyboard.git
cd predictive-keyboard

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the application
python app.py

📱 Demo
(You can add your Gradio public link here after running)
📊 Model Details

Architecture: Embedding → LSTM → LSTM → Dense
Sequence Length: 12 words
Vocabulary Size: ~12,000 words
Training Data: Indian English + Hinglish conversations

📁 Project Structure
textpredictive-keyboard/
├── model.h5                 # Trained LSTM model
├── tokenizer.pkl            # Tokenizer for text processing
├── app.py                   # Gradio Web Interface
├── requirements.txt         # Dependencies
└── README.md

🎯 Future Improvements

Fine-tuning with GPT-2 or larger models
Support for more Indian languages (Hindi, Marathi, etc.)
Mobile-friendly deployment
Personalization based on user typing style

Made by: PARTH KOKIL
