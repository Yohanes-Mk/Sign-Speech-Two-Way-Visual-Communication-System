import os  #makes it easier to navigate btw 2 file systems e.g windows and linux machine
import cv2 #to allow us use openCV
import tensorflow as tf
import numpy as np #good to have incase you need to preprocess any arrays.
from typing import List
from matplotlib import pyplot as plt
import imageio #allow us to convert a numpy array to a gif



def load_data(path: str): #it takes in the path to our video, we then split it and convert it so we have a video_path and alignemnbt_path.
    path = bytes.decode(path.numpy())
    file_name = path.split('/')[-1].split('.')[0] 
    # File name splitting for windows
    # file_name = path.split('\\')[-1].split('.')[0]
    video_path = os.path.join('..','data','movies',f'{file_name}.MP4')
    # alignment_path = os.path.join('..','data','alignments','s1',f'{file_name}.align')
    frames = load_video(video_path) 
    # alignments = load_alignments(alignment_path)
    
    return frames #then we return the frames and the alignement from each of the function.

def load_video(path:str) -> List[float]: #the function (load_video) takes a datapath and output a list of floats which will be representing the video

    cap = cv2.VideoCapture(path)#looping through all the videos
    frames = [] #Loop through each frames and store it inside an array called frames
    for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))): 
        ret, frame = cap.read()
        frame = tf.image.rgb_to_grayscale(frame) # converting it from rgb to greyscale so we will have lesser data to process
        frames.append(frame[190:236,80:220,:])  #This part here basically isolates the video to the lip region only.
    cap.release()
    
    mean = tf.math.reduce_mean(frames) #then we calculate the mean
    std = tf.math.reduce_std(tf.cast(frames, tf.float32)) #then we calculate the standard deviation, this is just good practice to scale the data
    return tf.cast((frames - mean), tf.float32) / std #then we scale a particular image feature


vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "] 
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)

# def load_alignments(path:str) -> List[str]: #a function used to load our alignment, it takess a specific path and use it to map through the alignement
#     with open(path, 'r') as f: #we open up the path
#         lines = f.readlines() #Here we read the contents of the files
#     tokens = []
#     for line in lines:
#         line = line.split()  #we split up each one of the lines
#         if line[2] != 'sil':  #if the line contain the value "sil" then we ignore it cause we dont really need it.
#             tokens = [*tokens,' ',line[2]]#append them into an array called tokens
#     return char_to_num(tf.reshape(tf.strings.unicode_split(tokens, input_encoding='UTF-8'), (-1)))[1:] #then we convert them from characters to num

