# pip3 install opencv-python==4.6.0.66 imageio==2.23.0 matplotlib==3.6.2

import os  #makes it easier to navigate btw 2 file systems e.g windows and linux machine
import cv2 #to allow us use openCV
import tensorflow as tf
import numpy as np #good to have incase you need to preprocess any arrays.
from typing import List
from matplotlib import pyplot as plt
import imageio #allow us to convert a numpy array to a gif



# 1. Build Data Loading Functions

#A preprocessing function that allows us to load the video
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

# Define our vocalbulary
vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "] #This is basically every single character we might expect in the training process

char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")  #Used to convert characters to numbers (encode)
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)#Used to convert numbers to characters (decode)

# print(
#     f"The vocabulary is: {char_to_num.get_vocabulary()} "
#     f"(size ={char_to_num.vocabulary_size()})"
# )

# char_to_num.get_vocabulary()


# A load alignemnt function
def load_alignments(path:str) -> List[str]: #a function used to load our alignment, it takess a specific path and use it to map through the alignement
    with open(path, 'r') as f: #we open up the path
        lines = f.readlines() #Here we read the contents of the files
    tokens = []
    for line in lines:
        line = line.split()  #we split up each one of the lines
        if line[2] != 'sil':  #if the line contain the value "sil" then we ignore it cause we dont really need it.
            tokens = [*tokens,' ',line[2]]#append them into an array called tokens
    return char_to_num(tf.reshape(tf.strings.unicode_split(tokens, input_encoding='UTF-8'), (-1)))[1:] #then we convert them from characters to num


# A load data function, This is to load the alignment and the video simultanouesly
def load_data(path: str): #it takes in the path to our video, we then split it and convert it so we have a video_path and alignemnbt_path.
    path = bytes.decode(path.numpy())
    file_name = path.split('/')[-1].split('.')[0] 
    # File name splitting for windows
    # file_name = path.split('\\')[-1].split('.')[0]
    video_path = os.path.join('data','s1',f'{file_name}.mpg')
    alignment_path = os.path.join('data','alignments','s1',f'{file_name}.align')
    frames = load_video(video_path) 
    alignments = load_alignments(alignment_path)
    
    return frames, alignments #then we return the frames and the alignement from each of the function.

#testing it out
# test_path = '.\\data\\s1\\bbal6n.mpg'
test_path='./data/s1/bbal6n.mpg'


tf.convert_to_tensor(test_path).numpy().decode('utf-8').split('\\')[-1].split('.')[0]
frames, alignments = load_data(tf.convert_to_tensor(test_path))
#Returning back a bunch of frames which shows the person mouth moving
# plt.imshow(frames[40]) #this is to show the image of the frame

# plt.imshow(frames[40])  # Assuming grayscale image
# plt.show()

#converting the alignement into a coded sequence which the machine will now understand.
# print (alignments)


tf.strings.reduce_join([bytes.decode(x) for x in num_to_char(alignments.numpy()).numpy()])
# this is the word representation of what is being said


#It will be used in the data pipeline, it is basically wrapping our load_data function inside a tf.py_function and allowing us to work with a specific file_path format

def mappable_function(path:str) ->List[str]:
    result = tf.py_function(load_data, [path], (tf.float32, tf.int64)) #wrapping the load data in a mappable function ehich allows us to return float32 and int64 
    return result





# 2. Create Data Pipeline
#Creating our dataset 
data = tf.data.Dataset.list_files('./data/s1/*.mpg') #going to go into our data folder,into the s1 folder and look for anything that ends with .mpg
data = data.shuffle(500, reshuffle_each_iteration=False)  #we dont want to reshuffle after each line of iteration, getting the first 500.
data = data.map(mappable_function)#mapping through the data
data = data.padded_batch(2, padded_shapes=([75,None,None,None],[40])) #putting them into a batch so each of them will have 2videos and 2sets of alignments
data = data.prefetch(tf.data.AUTOTUNE) #prefetching to make sure we optimize our data pipeline, so we are preloading as our machine model is still learning 
# Added for split 
train = data.take(450) #creating a training partition by taking the first 450 samples 
test = data.skip(450)  #Our testing portion is anything after the 450


# Training  the dataset
frames, alignments = data.as_numpy_iterator().next()
sample = data.as_numpy_iterator()
val = sample.next(); val[0]
# plt.imshow(val[0][0][35])
# plt.show


# This is the processed annotation
champ = tf.strings.reduce_join([num_to_char(word) for word in val[1][0]])
#This is for the alignment, we are looping through every word in the alignment
# print(champ)



# 3. Design the Deep Neural Network
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv3D, LSTM, Dense, Dropout, Bidirectional, MaxPool3D, Activation, Reshape, SpatialDropout3D, BatchNormalization, TimeDistributed, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler


# data.as_numpy_iterator().next()[0][0].shape
# print(data.as_numpy_iterator().next()[0][0].shape)
# CNNs Convolutional Neural Networks
# Conv3D = A CNN with 3D layer dimension

#Here we define the deep neural network
model = Sequential() #we are initializing  the model
model.add(Conv3D(128, 3, input_shape=(75,46,140,1), padding='same')) # the input shape was what we got from our data in line 164,we will have a 128 3D convolution unit, and will be 3by3 in size, the padding:same is to preserve the shape of our input.
model.add(Activation('relu')) # To give us a non-linearity to our neural network
model.add(MaxPool3D((1,2,2))) # We are condensing it down here, it takes the max value from each frame and condense it down to a 2x2 square

model.add(Conv3D(256, 3, padding='same')) #we will have a 256 3D convolution unit
model.add(Activation('relu')) #Activation (ReLU): to introduce non-linearity.
model.add(MaxPool3D((1,2,2)))

model.add(Conv3D(75, 3, padding='same')) #we will have a 75 3D convolution unit
model.add(Activation('relu'))
model.add(MaxPool3D((1,2,2)))

model.add(TimeDistributed(Flatten())) #this  allows us to have 75 inputs into our LSTM so we can output 75units representing the text based characters 

model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True))) #(Bidirectional(LSTM)) are employed to process sequences in both forward and backward directions.
model.add(Dropout(.5)) #Dropouts are added to prevent overfitting.

model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
model.add(Dropout(.5))

model.add(Dense(char_to_num.vocabulary_size()+1, kernel_initializer='he_normal', activation='softmax')) #this takes our vocab-size +1 
# model.summary()


#Testing the model
# yhat = model.predict(val[0]) #passing the sample data to the model
# print(yhat[0].shape) # printing out the raw value
# print(tf.strings.reduce_join([num_to_char(tf.argmax(x)) for x in yhat[1]])) #printing the prediction result
# print(model.output_shape)




# 4. Setup Training Options and Train

#We are defining a learning rate scheduler
def scheduler(epoch, lr):
    if epoch < 30: #if our learning rate(lr) is below 30
        return lr
    else:
        return lr * tf.math.exp(-0.1) #if its not, we will drop it down.
    
#We are defining our CTCLoss : Connectionist Temporal Classification Loss
def CTCLoss(y_true, y_pred):
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64") #We are taking the batch length
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64") #We are calculating the input length
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64") # WE are calculating the label length

    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

    loss = tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)  #We are passing the y_true value, y_pred value, inputLength which is 75,labelLength which is 40, through tf.keras.backend.ctc_batch_cost which returns the value of CTCLoss 
    return loss

class ProduceExample(tf.keras.callbacks.Callback): 
    def __init__(self, dataset) -> None: 
        self.dataset = dataset.as_numpy_iterator()
    
    def on_epoch_end(self, epoch, logs=None) -> None:
        data = self.dataset.next()
        yhat = self.model.predict(data[0])
        decoded = tf.keras.backend.ctc_decode(yhat, [75,75], greedy=False)[0][0].numpy()
        for x in range(len(yhat)):           
            print('Original:', tf.strings.reduce_join(num_to_char(data[1][x])).numpy().decode('utf-8')) #Here we are producing the origianl annotation 
            print('Prediction:', tf.strings.reduce_join(num_to_char(decoded[x])).numpy().decode('utf-8'))#Here we are producing the predicted annotation 
            print('~'*100)

model.compile(optimizer=Adam(learning_rate=0.0001), loss=CTCLoss) #This is to compile our model, we are setting our optimizer to an adam optimizer and definig our loss as the CTC loss
checkpoint_callback = ModelCheckpoint(os.path.join('models','checkpoint'), monitor='loss', save_weights_only=True) #This is to save our model checkpoint after every epoch, when our model trains it will be saved in a models folder and called checkpoint, we are monitoring our loss and saving our weight
schedule_callback = LearningRateScheduler(scheduler) #This is allows us to drop our learning rate  when we get to epoch 30
example_callback = ProduceExample(test)#This output our prediction after each epoch to see how well our model is making prediction
# model.fit(train, validation_data=test, epochs=100, callbacks=[checkpoint_callback, schedule_callback, example_callback])




# 5. Make a Prediction
model.load_weights('models/checkpoint') 
test_data = test.as_numpy_iterator() 
sample = test_data.next()  
yhat = model.predict(sample[0])  

print('~'*100, 'REAL TEXT')
print([tf.strings.reduce_join([num_to_char(word) for word in sentence]) for sentence in sample[1]]) 

decoded = tf.keras.backend.ctc_decode(yhat, input_length=[75,75], greedy=True)[0][0].numpy()
print('~'*100, 'PREDICTIONS')
print([tf.strings.reduce_join([num_to_char(word) for word in sentence]) for sentence in decoded])




# 6 Test on a Video
# sample = load_data(tf.convert_to_tensor('./data/s1/swwv9a.mpg'))
# print('/..this is before the real text')
# print('~'*50, 'REAL TEXT')
# print([tf.strings.reduce_join([num_to_char(word) for word in sentence]) for sentence in [sample[1]]])

# yhat = model.predict(tf.expand_dims(sample[0], axis=0))
# decoded = tf.keras.backend.ctc_decode(yhat, input_length=[75], greedy=True)[0][0].numpy()
# print('/..this is before the prediction')
# print('~'*50, 'PREDICTIONS')
# print([tf.strings.reduce_join([num_to_char(word) for word in sentence]) for sentence in decoded])

