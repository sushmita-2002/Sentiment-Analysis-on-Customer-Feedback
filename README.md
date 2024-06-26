# Sentiment-Analysis-on Customer Feedback

## Overview
This README provides details about the sentiment analysis models (LSTM and BERT) implemented in this project, along with instructions for setting up and running the frontend applications developed for both models.

## Table of Contents
- [Dataset](#Dataset)
- [Models](#Models)
- [Frontend](#Frontend)
- [Visual Overview](#Visual-Overview)
- [Dependencies](#Dependencies)
- [Usage](#Usage)

## Dataset 
Dataset Link : https://drive.google.com/drive/u/0/folders/1Xym22vc-guTMvPGI00vOUz_hszmB3dPU

## Models
### 1. LSTM Model
- **Description:** The LSTM (Long Short-Term Memory) model is a type of recurrent neural network (RNN) designed for sequence prediction tasks, making it suitable for sentiment analysis of text data.
- **Implementation:** The LSTM model is implemented using TensorFlow/Keras.
- **Usage:** The model takes text inputs and predicts sentiment labels (e.g., positive, negative, neutral).
### 2. BERT Model
- **Description:** BERT (Bidirectional Encoder Representations from Transformers) is a transformer-based language model developed by Google. The BERT model has demonstrated state-of-the-art performance in various NLP tasks, including sentiment analysis.
- **Implementation:** We utilize the Hugging Face transformers library to implement the BERT model.
- **Usage:** The BERT model accepts text inputs and outputs sentiment predictions with high accuracy.

## Frontend
### 1. Frontend for LSTM Model
- **Description:** We have developed a user-friendly web application interface for interacting with the LSTM sentiment analysis model.
- **Features:**
   - Input text box for entering the text to be analyzed.
   - Button to trigger the sentiment analysis process.
   - Display of the predicted sentiment label (e.g., positive, negative) based on the input text.
- **Setup:**
   1. Navigate to the frontend/lstm directory.
   2. Open index.html in a web browser.
### 2. Frontend for BERT Model
- **Description:** Similarly, we have created a frontend interface for interacting with the BERT sentiment analysis model.
- **Features:**
  - Text input field for entering the text to be analyzed.
  - Button to initiate the sentiment analysis.
  - Display of the predicted sentiment label generated by the BERT model.
- **Setup:**
  1. Go to the frontend/bert directory.
   2. Launch index.html in a web browser.

## Visual Overview
- Test with positve review:
  
  ![positive review](https://github.com/sushmita-2002/Sentiment-Analysis-Model/assets/80975689/50a965d8-cc9e-43dc-a917-5d556f1ec8b6)

  ![positive review output](https://github.com/sushmita-2002/Sentiment-Analysis-Model/assets/80975689/b60bd774-a387-401e-82c4-1c0c6aabcd30)

- Test with negative review:

  ![negative review](https://github.com/sushmita-2002/Sentiment-Analysis-Model/assets/80975689/f171974f-bd88-4cad-874c-3cdc955c2664)
  
  ![negative review output](https://github.com/sushmita-2002/Sentiment-Analysis-Model/assets/80975689/e9e12bd4-5d14-4cad-9523-ea2040979c87)


## Dependencies
- Python 3.x
- TensorFlow (for LSTM model)
- Hugging Face transformers library (for BERT model)
- Web browser (for frontend applications)
  
## Usage
### Running the Models
1. Ensure all necessary Python dependencies are installed (tensorflow, transformers).
2. Execute the respective model scripts (lstm_model.py, bert_model.py) to load and initialize the models.
### Using the Frontend
1. Open the frontend application (index.html) for the desired model in a web browser.
2. Enter text into the input field.
3. Click the button to perform sentiment analysis.
4. View the predicted sentiment label displayed on the webpage.


