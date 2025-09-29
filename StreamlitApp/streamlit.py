# Import all of the dependencies 
import streamlit as st
import os
import imageio
import tensorflow as tf 
from PIL import Image

from utils import  load_data, num_to_char 
from model import load_model 

  
# Set the layout to the streamlit app as wide 
st.set_page_config(layout='wide')

with st.sidebar: #this is to create sidebar
    # st.image('..','profilpic.jpeg')
    st.title('LipBuddy')
    st.info('This application is originally developed from the LipNet deep learning model.')


# st.title('LipNet Full Stack App') 
# Generating a list/array of videos option 
options = os.listdir(os.path.join('..','data', 'movies'))   
selected_video = st.selectbox('Choose video', options)


# Generate two columns 
col1, col2 = st.columns(2)
Loaded = False

if options:

    # Rendering the video 
    with col1: 
        st.info('The video below displays the converted video in mp4 format')
        file_path = os.path.join('..','data','movies', selected_video) #to get the path of the selected video
        os.system(f'ffmpeg -i {file_path} -vcodec libx264 test_video.mp4 -y') #to change the format of the selected video from .mpg to .mp4

        # Rendering inside of the app
        video = open('test_video.mp4', 'rb')
        # st.write(file_path)
        # video = open(file_path, 'rb') 
        video_bytes = video.read() 
        st.video(video_bytes)


    with col2:

        video = load_data(tf.convert_to_tensor(file_path))

        # Process the video in overlapping segments of 75 frames
        segment_size = 75
        stride = 25  # Adjust as needed
        num_frames = len(video)


        full_decoded_text = ""  # To accumulate decoded text across segments
        full_tokens=[]

        for start in range(0, num_frames, stride):
            end = start + segment_size
            segment = video[start:end]

            # Ensure the segment has exactly 75 frames
            segment = tf.concat([segment, tf.zeros((75 - segment.shape[0], segment.shape[1], segment.shape[2], segment.shape[3]))], axis=0)

            model = load_model()

            # Adjust the input shape to match the model's expectation
            processed_frames = tf.expand_dims(segment, axis=0)
            processed_frames = processed_frames[:, :75, :, :, :]  # Trim frames to match expected number

            yhat = model.predict(processed_frames)

            decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
            full_tokens.append(decoder)

            converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
            full_decoded_text += converted_prediction  # Accumulate decoded text

        st.info('This is the output of the machine learning model as tokens')
        st.text(decoder)

        # Display the accumulated decoded text on the same line
        st.info('This is the Prediction')
        st.text(full_decoded_text)


        


  
      
