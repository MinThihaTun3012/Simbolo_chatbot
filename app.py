import numpy as np
import pandas as pd
import tensorflow as tf
import nltk
import json
import random
import streamlit as st
import pickle
import scikeras
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from nltk.stem import WordNetLemmatizer
from keras.optimizers import SGD


nltk.download('wordnet')



# <-------- load necessary items -------------------->
# load the json file
with open('intents.json', 'rb') as d:
    data_set = json.load(d)
     
# load words and labels
with open("pickle_files/data.pkl","rb") as f:
    data = pickle.load(f)
    words = data["words"]
    labels = data["labels"]

#load model
with open("pickle_files/model.pkl", "rb") as f:
    model = pickle.load(f)
    
# load lemmatizer
with open("pickle_files/stem.pkl", "rb") as f:
    lemmatizer = pickle.load(f)
# <----------------end line of loading ----------------->    


# <----------start of function for predicting-------------------->

# function for making response according threshold
def  make_response(result, threshold):
    
    
    result_index = np.argmax(result)
    tag = labels[int(result_index)]
    
    if result[0][result_index] > threshold: 
        for tg in data_set["intents"]:
            if tg["tag"] == tag:
                response = random.choice(tg["responses"])
                break
    else:
        response = "I don't understand. Please can you ask me?"


    return response
    
# function of bag of words
def bag_of_words(s):
    
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [lemmatizer.lemmatize(w.lower()) for w in s_words if w not in ["?", "."]]

    for s_word in s_words:
        for i,w in enumerate(words):
            if w== s_word:
                bag[i] += 1 
    return np.array(bag).reshape(1,-1)

# Function for predicting 
def model_predict(s):
    
    result = model.predict([bag_of_words(s)])
    response = make_response(result, 0.70)
    max = {result[0][np.argmax(result)]}

    return response, max
    
# <------- end line of function for predicting ----------->


# <----------start of app------------> 

# Set page config
st.set_page_config(
    page_title = "Simbolo Chatbot",
    page_icon = "image/smboloitschool_logo.jpg",
    layout="centered", 
    initial_sidebar_state="expanded"
)

# Display header
st.title("Simbolo Chat Bot")

# Disclaimer expandable text
with st.expander("❗Disclaimer"):
    st.markdown(
        "* Model may produce wrong responses or wrong data.\n"
        "* This app is for educational purposes only.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    if message["image"] == 1:
        with st.chat_message(message["role"], avatar="image/smboloitschool_logo.jpg"):
            d_max = message["max"] 
            d_content = message["content"]
            st.markdown(f"max probability{d_max} \n \n {d_content}")
    else:
        with st.chat_message(message["role"]): 
            st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is up?"):
    
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
        
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt, "image": 0})

    with st.chat_message("assistant", avatar="image/smboloitschool_logo.jpg"): 
        response, max = model_predict(prompt)
        st.markdown(f"max probability : {max}  \n  \n{response}")

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response, "image":1, "max":max})
    
# <----------end of app------------>
