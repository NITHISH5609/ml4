import os
from PIL import Image
import keras
from keras.preprocessing.text import tokenizer_from_json
import captioning
import json
import streamlit as st
import translate

st.title("Dekh.ai")
st.header("Image Captioning Translator")

with open('models/tokenizer.json', 'r') as f:
    tokenizer_json = json.load(f)
tokenizer = tokenizer_from_json(tokenizer_json)
    
model = keras.models.load_model("models/image_captioning.h5")
vocab_size = tokenizer.num_words
max_length = 37

uploaded_file = st.file_uploader(label="Upload an Image")
if uploaded_file:
    img = Image.open(uploaded_file)
    
    new_folder = os.path.join(os.getcwd(), 'uploaded_images')
    if not os.path.exists(new_folder):
        os.makedirs(new_folder)
    
    image_path = os.path.join(new_folder, uploaded_file.name)
    img.save(image_path)
    
    features = captioning.image_features("uploaded_images")
    imgname,_ = uploaded_file.name.split('.')
    caption = captioning.give_caption(model, tokenizer, max_length, features[imgname])
    
    st.image(f"uploaded_images/{uploaded_file.name}")
    st.write(caption)
    choice = st.radio("Select Target Text",("hi","kn","ta"))
    st.write(translate.caption_translate("en",choice,caption))