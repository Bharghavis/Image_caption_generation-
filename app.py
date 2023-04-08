import streamlit as st
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import re
import pickle

# Load the pre-trained InceptionV3 model
image_model = InceptionV3(include_top=True)
# Remove the last layer of the model
feat_extractor = Model(inputs=image_model.input, outputs=image_model.layers[-2].output)

# Load the pre-trained caption generation model
model = load_model('caption_generator.h5')

# Load the tokenizer for the model
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Set maximum length for caption generation
max_length = 34

# Define function to preprocess input image
def preprocess_image(img_path):
    img = load_img(img_path, target_size=(299, 299))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 127.5 - 1.
    return img_array

# Define function to generate caption for input image
def generate_caption(img_path):
    # Preprocess the image
    img_array = preprocess_image(img_path)
    # Extract features from the image using the pre-trained InceptionV3 model
    feat_vector = feat_extractor.predict(img_array)
    # Generate caption for the image using the pre-trained caption generation model
    input_seq = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([input_seq])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        prediction = model.predict([feat_vector, sequence], verbose=0)
        pred_word = ''
        for word, index in tokenizer.word_index.items():
            if index == np.argmax(prediction):
                pred_word = word
                break
        input_seq += ' ' + pred_word
        if pred_word == 'endseq':
            break
    caption = input_seq.split()[1:-1]
    caption = ' '.join(caption)
    caption = re.sub(r'\b(\w+)( \1\b)+', r'\1', caption)
    return caption

# Define Streamlit app
def app():
    st.title('Image Caption Generator')
    # Allow user to upload an image file
    uploaded_file = st.file_uploader('Choose an image file', type=['jpg', 'jpeg', 'png'])
    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption='Uploaded image', use_column_width=True)
        # Generate caption for the uploaded image
        caption = generate_caption(uploaded_file)
        st.write('**Caption:**', caption)

# Run the Streamlit app
if __name__ == '__main__':
    app()
