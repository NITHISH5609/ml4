import os
import numpy as np
import keras
import tensorflow as tf

def image_features(directory):
    model = tf.keras.applications.vgg16.VGG16()
    model = keras.models.Model(inputs=model.input, outputs=model.layers[-2].output) #Remove the final layer
    
    features = {}
    for f in os.listdir(directory):
        filename = directory + "/" + f
        identifier = f.split('.')[0]
        
        image = keras.utils.load_img(filename, target_size=(224,224))
        arr = keras.utils.img_to_array(image, dtype=np.float32)
        arr = arr.reshape((1, arr.shape[0], arr.shape[1], arr.shape[2]))
        arr = keras.applications.vgg16.preprocess_input(arr)
    
        feature = model.predict(arr, verbose=0)
        features[identifier] = feature
    return(features)

def give_caption(model, tokenizer, max_length, feature):
    caption = "<startseq>"
    while True:
        encoded = tokenizer.texts_to_sequences([caption])[0]
        padded = keras.utils.pad_sequences([encoded], maxlen=max_length, padding='pre')[0]
        padded = padded.reshape((1, max_length))
        
        pred_Y = model.predict([feature, padded])[0,-1,:]
        next_word = tokenizer.index_word[pred_Y.argmax()]
        caption = caption + ' ' + next_word
        
        if next_word == '<endseq>' or len(caption.split()) >= max_length:
            break
        
    caption = caption.replace('<startseq> ', '')
    caption = caption.replace(' <endseq>', '')
    
    return(caption)