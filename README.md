# Sentiment-Analysis-Model

##Overview
This README provides details about the sentiment analysis models (LSTM and BERT) implemented in this project, along with instructions for setting up and running the frontend applications developed for both models.

##Dataset 
link : https://drive.google.com/drive/u/0/folders/1Xym22vc-guTMvPGI00vOUz_hszmB3dPU

##Models
1. LSTM Model
- Description: The LSTM (Long Short-Term Memory) model is a type of recurrent neural network (RNN) designed for sequence prediction tasks, making it suitable for sentiment analysis of text data.
- Implementation: The LSTM model is implemented using TensorFlow/Keras.
- Usage: The model takes text inputs and predicts sentiment labels (e.g., positive, negative, neutral).
2. BERT Model
Description: BERT (Bidirectional Encoder Representations from Transformers) is a transformer-based language model developed by Google. The BERT model has demonstrated state-of-the-art performance in various NLP tasks, including sentiment analysis.
Implementation: We utilize the Hugging Face transformers library to implement the BERT model.
Usage: The BERT model accepts text inputs and outputs sentiment predictions with high accuracy.
Frontend Applications
Frontend for LSTM Model
Description: We have developed a user-friendly web application interface for interacting with the LSTM sentiment analysis model.
Features:
Input text box for entering the text to be analyzed.
Button to trigger the sentiment analysis process.
Display of the predicted sentiment label (e.g., positive, negative) based on the input text.
Setup:
Navigate to the frontend/lstm directory.
Open index.html in a web browser.
Frontend for BERT Model
Description: Similarly, we have created a frontend interface for interacting with the BERT sentiment analysis model.
Features:
Text input field for entering the text to be analyzed.
Button to initiate the sentiment analysis.
Display of the predicted sentiment label generated by the BERT model.
Setup:
Go to the frontend/bert directory.
Launch index.html in a web browser.
Dependencies
Python 3.x
TensorFlow (for LSTM model)
Hugging Face transformers library (for BERT model)
Web browser (for frontend applications)
Usage
Running the Models
Ensure all necessary Python dependencies are installed (tensorflow, transformers).
Execute the respective model scripts (lstm_model.py, bert_model.py) to load and initialize the models.
Using the Frontend
Open the frontend application (index.html) for the desired model in a web browser.
Enter text into the input field.
Click the button to perform sentiment analysis.
View the predicted sentiment label displayed on the webpage.
Notes
The LSTM and BERT models should be trained on appropriate sentiment analysis datasets before deployment.
Ensure that the frontend applications (index.html files) have necessary CORS (Cross-Origin Resource Sharing) permissions set if interacting with models hosted on a different server.
