import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

class Dataset:
    """
    Creates a tensorflow dataset for video files.
    The data needs to been in a Imagenet directory structure.
    After processing the data, two tf.data train and test will be returned.
    
    Parameters: 
        data_path: A string of the parent directory for all the data.
        class_list: A list containing classes names that will be needed in the dataset.
        seq_len(default=20): A integer for selecting total frames from the video.
        frame_size(default=128): A integer for resizing height and width of the frames.
        batch_size(default=32): A integer for selecting the size of a batch.
        seed(default=42): A integer for controlling the randomness of random numbers generator.
    
    Returns:
        train_ds, test_ds: A tuple of training and testing dataset pipeline. 
    """
    def __init__(self, data_path, seq_len = 20, frame_size = 128, batch_size = 32, seed = 42, class_list=None):
        self.data_path = data_path
        self.seq_len = seq_len
        self.frame_size = frame_size
        self.batch_size = batch_size
        self.seed = seed
        
        # Handling class_list
        if class_list==None:
            self.classes = sorted(os.listdir(self.data_path))
        else:
            self.classes = sorted(class_list)
    
    def frames_extraction(self, video_file_path):
        frames_list = []
        # Reading the video file and counting the frames
        video_reader = cv2.VideoCapture(video_file_path)
        video_frame_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))

        # Selecting the frames at certain interval and applying the transformation
        skip_frames = max(int(video_frame_count/self.seq_len), 1)
        for i in range(self.seq_len):
            video_reader.set(cv2.CAP_PROP_POS_FRAMES, i * skip_frames)
            success, frame = video_reader.read()
            if not success:
                break
            resize_frame = cv2.resize(frame, (self.frame_size, self.frame_size))
            norm_frame = resize_frame/255.
            frames_list.append(norm_frame.astype('float32'))
        video_reader.release()
        return frames_list
    
    def create_dataset(self):
        features = []
        labels = []
        video_files_path = []
        
        # Going through all the data in the class list
        print(f'[INFO] Extracting data from {len(self.classes)} classes...')
        for i, class_name in enumerate(self.classes):
            print(f'[INFO] Extracting all the data in the class: {class_name}')

            # Getting the list of all the video files and the path to the video
            files_list = os.listdir(os.path.join(self.data_path, class_name))
            for file_name in files_list:
                video_file_path = os.path.join(self.data_path, class_name, file_name)

                # Extracting frames using the function and verifying the total frames
                frames_list = self.frames_extraction(video_file_path=video_file_path)
                if len(frames_list) == self.seq_len:

                    # Appending the data in a list
                    features.append(frames_list)
                    labels.append(i)
                    video_files_path.append(video_file_path)
        
        # Converting the list to array
        features = np.asarray(features)
        labels = np.asarray(labels)
        
        # Hot encoding the labels
        labels = to_categorical(labels)
        print('[INFO] Datset is been created')
        return features, labels, video_files_path
    
    def split_dataset(self):
        features, labels, video_files_path = self.create_dataset()
        train_features, test_features, train_labels, test_labels = train_test_split(features, 
                                                                                    labels, 
                                                                                    test_size=0.25, 
                                                                                    shuffle=True, 
                                                                                    random_state=self.seed)
        print('[INFO] Dataset is been splitted into train and test set.')
        return train_features, test_features, train_labels, test_labels
    
    def dataset_pipeline(self):
        train_features, test_features, train_labels, test_labels = self.split_dataset()
        train_ds = tf.data.Dataset.from_tensor_slices((train_features,
                                                       train_labels)).shuffle(10000, self.seed).batch(self.batch_size, True).prefetch(tf.data.AUTOTUNE)
        test_ds = tf.data.Dataset.from_tensor_slices((test_features,
                                                      test_labels)).batch(self.batch_size, True).prefetch(tf.data.AUTOTUNE)
        print('[INFO] Dataset pipeline is been created')
        return train_ds, test_ds
