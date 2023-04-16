import os
import cv2
import argparse
import numpy as np
from sys import exit
import tensorflow as tf

# Creating a parser
parser = argparse.ArgumentParser(description='Get some hyperparameters')

# Getting hyperparameters
# Sequence length
parser.add_argument('--seq_len',
                    default=20,
                    type=int,
                    help='Total number of frames for every video')

# Frame size
parser.add_argument('--frame_size',
                    default=128,
                    type=int,
                    help='Integer for resizing the frame')

# Video file path
parser.add_argument('--video_path',
                    type=str,
                    help='File path of the video for predicting.')

# Saved model
parser.add_argument('--model_path',
                    default='saved_model/lrcn_model2_2023-03-21-09:40:09_loss:0.1253_accuracy:0.9622',
                    type=str,
                    help='Target model path to use for the prediction.')

# Class list
parser.add_argument('--class_list',
                    default=['Biking', 'Diving', 'GolfSwing', 'Punch', 'Rowing'],
                    nargs='+',
                    help='A list containing class names')

args = parser.parse_args()

SEQ_LEN = args.seq_len
FRAME_SIZE = args.frame_size
VIDEO_PATH = args.video_path
MODEL_PATH = args.model_path
CLASS_LIST = args.class_list

print(f'[INFO] Predicting video file: "{VIDEO_PATH}" using model: "{MODEL_PATH}".')

# Loading the model
model = tf.keras.models.load_model(filepath=MODEL_PATH)
print('[INFO] Model is been loaded and ready for prediction.')

# function to process the video file
def frames_extraction(video_file_path):
    frames_list = []
    
    # Reading the video file and counting the frames
    video_reader = cv2.VideoCapture(video_file_path)
    video_frame_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Selecting the frames at certain interval and applying the transformation
    skip_frames = max(int(video_frame_count/SEQ_LEN), 1)
    for i in range(SEQ_LEN):
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, i * skip_frames)
        success, frame = video_reader.read()
        if not success:
            break
        resize_frame = cv2.resize(frame, (FRAME_SIZE, FRAME_SIZE))
        norm_frame = resize_frame/255.
        frames_list.append(norm_frame.astype('float32'))
    video_reader.release()
    return frames_list

# Processing the video file
frames_list = frames_extraction(video_file_path=VIDEO_PATH)
print('[INFO] Video file is ready for prediction.')

# predicting using the model 
pred_prob = model.predict(np.expand_dims(frames_list, axis=0))[0]
pred_label = np.argmax(pred_prob)
pred_class = args.class_list[pred_label]

# printing the result
print(f'[INFO] Action predicted by the model : {pred_class}')
print(f'[INFO] Prediction probalities: {pred_prob[pred_label]:.2f}\n')
