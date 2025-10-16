# Import all of the dependencies 
import streamlit as st
import os
import imageio
import tensorflow as tf 
import subprocess

from main import  load_data, num_to_char 
from modelutil import load_model 

  
# Set the layout to the streamlit app as wide 
st.set_page_config(layout='wide')

with st.sidebar: #this is to create sidebar
    st.image('profilpic.jpeg')
    st.title('LipBuddy')
    st.info('This application is originally developed from the LipNet deep learning model.')


# st.title('LipNet Full Stack App') 
# Generating a list/array of videos option 
options = os.listdir(os.path.join( 'data', 's1'))   
selected_video = st.selectbox('Choose video', options)


# Generate two columns 
col1, col2 = st.columns(2)


if options: 

    # Rendering the video 
    with col1: 
        st.info('The video below displays the converted video in mp4 format')
        file_path = os.path.join('data','s1', selected_video) #to get th epath of the selected video
        os.system(f'ffmpeg -i {file_path} -vcodec libx264 test_video.mp4 -y') #to change the format of the selected video from .mpg to .mp4

        # Rendering inside of the app
        video = open('test_video.mp4', 'rb') 
        video_bytes = video.read() 
        st.video(video_bytes)


    with col2: 
        st.info('This is all the machine learning model sees when making a prediction')
        video, annotations = load_data(tf.convert_to_tensor(file_path))
        imageio.mimsave('animation.gif', video, fps=10)
        st.image('animation.gif', width=400) 

        st.info('This is the output of the machine learning model as tokens')
        model = load_model()
        yhat = model.predict(tf.expand_dims(video, axis=0))
        decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
        st.text(decoder)

        # Convert prediction to text
        st.info('Decode the raw tokens into words')
        converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
        st.text(converted_prediction)


        # For the text to speech
        st.info('Convert the text to speech')
        text = converted_prediction
        audio_file_path = "output.wav"  # Save the audio to a file
        subprocess.run(["espeak", "-w", audio_file_path,"-v", "en-us", "-p", "50", "-s", "150", "-g", "5", text])  # Use -w to write to a file
        st.audio(audio_file_path, format='audio/wav')  # Display the audio file
