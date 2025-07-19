#  Web app to predict the next word in a sentence using a trained LSTM model using Streamlit


import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
import pickle


st.title("Next Word Predictor")
#  show loading spinner while loading the model and tokenizer

with st.spinner("Loading model and tokenizer..."):

    # Load the tokenizer
    with open("tokenizer.pkl", "rb") as handle:
        tokenizer = pickle.load(handle)

    # Load the trained model
    model = load_model("lstm_text_prediction_model.h5")

    #  stop the spinner


#  now predict the next word
def predict_next_word(model, tokenizer, text, max_sequence_length):
    # Tokenize the input text
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_length:
        # If the input text is longer than the max sequence length, truncate it
        token_list = token_list[-(max_sequence_length - 1) :]
    # Pad the sequence to the maximum length
    token_list = pad_sequences(
        [token_list], maxlen=max_sequence_length - 1, padding="pre"
    )
    # Predict the next word
    predicted = model.predict(token_list, verbose=0)
    # Get the index of the highest probability word
    predicted_index = np.argmax(predicted, axis=-1)[0]
    #  check if the predicted index is in the tokenizer's index_word
    if predicted_index in tokenizer.index_word:
        # Get the word corresponding to the index
        predicted_word = tokenizer.index_word[predicted_index]
        return predicted_word
    else:
        return None


# It has a title and a text input for the user to enter a sentence, and a button to predict the next word.
text_input = st.text_input(label="Enter a sentence:")
if st.button("Predict Next Word"):
    max_sequence_length = (
        model.input_shape[1] + 1
    )  # Get the maximum sequence length from the model's input shape
    predicted_word = predict_next_word(
        model, tokenizer, text_input, max_sequence_length
    )
    if predicted_word:
        st.write(f"Next word: '{predicted_word}'")
    else:
        st.write("No prediction available for the given input.")
else:
    st.write("Enter a sentence and click the button to predict the next word.")
