from flask import Flask, render_template, request
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import os
import pickle

# Set up the application and other configurations
app = Flask(__name__)
IMAGE_FOLDER = os.path.join('static', 'img_pool')
app.config['UPLOAD_FOLDER'] = IMAGE_FOLDER

# Load model and prepare TensorFlow graph if necessary
def init():
    global model, tokenizer
    model = load_model('sentiment_analysis.h5')
    # Load the tokenizer
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    global max_features  # Define max_features globally
    max_features = 1000  # The model's vocabulary size, set this as per your model's configuration
                                                                                                                            
@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template("home.html")

@app.route('/sentiment_analysis_prediction', methods=['POST'])
def sent_anly_prediction():
    try:
        if request.method == 'POST':
            text = request.form['text']
            x_test = tokenizer.texts_to_sequences([text.lower()])  # Tokenize text
            x_test = pad_sequences(x_test, maxlen=100)  # Pad sequence

            predictions = model.predict(x_test)
            print("Predictions: ", predictions)
            print("Processed text sequence: ", x_test)
            class_label = (predictions > 0.5).astype(int)  # Since you use sigmoid, directly threshold the output
            print(class_label)
            sentiment = 'Positive' if class_label else 'Negative'
            probability = float(predictions[0, 0])  # Get the sigmoid output as probability
            if sentiment == 'Positive':
                img_filename = 'Smiling_Emoji.jpeg'
            elif sentiment == 'Negative':
                img_filename = 'Sad_Emoji.jpeg'


            return render_template('index.html', text=text, sentiment=sentiment, probability="{:.2f}".format(probability), image=img_filename)
    except Exception as e:
        return str(e)  # For debugging purposes, display the error message on the web page



if __name__ == "__main__":
    init()
    app.run(debug=True)
