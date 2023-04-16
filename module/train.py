import os
import argparse
import datetime as dt
from sys import exit
import data_setup, model_builder, utils
import tensorflow as tf

# Creating a parser
parser = argparse.ArgumentParser(description='Get some hyperparameters')

# Getting hyperparameters
# Model name
parser.add_argument('--model_name',
                    default='LRCN',
                    choices=('ConvLSTM', 'LRCN'),
                    type=str,
                    help='Name of the model')

# Experiment name
parser.add_argument('--exp_name',
                    default='experiment',
                    type=str,
                    help='Name of the experiment')

# Number of epochs
parser.add_argument('--num_epochs',
                    default=100,
                    type=int,
                    help='The number of epochs to train the model')

# batch size
parser.add_argument('--batch_size',
                    default=32,
                    type=int,
                    help='The number of sample for batch data')

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

# learning rate
parser.add_argument('--lr',
                    default=0.001,
                    type=float,
                    help='Learning rate for optimizer')

# Data directory path
parser.add_argument('--data_dir_path',
                    default='data/UCF50',
                    type=str,
                    help='A path for data directory')

# Classes list
parser.add_argument('--class_list',
                    default=None,
                    nargs='+',
                    help='A list containing class names, use None for all the classes')

# Number of workers
parser.add_argument('--num_workers',
                    default=os.cpu_count(),
                    type=int,
                    help='Workers you want to assign durning the model training.')

# Callbacks
parser.add_argument('--callbacks',
                    default='True',
                    choices=('True', 'False'),
                    type=str,
                    help='Select a boolean to use callbacks durning training.')

# Getting the arguments from parser 
args = parser.parse_args()

# Collecting the arguments
MODEL_NAME = args.model_name
EXP_NAME = args.exp_name
NUM_EPOCHS = args.num_epochs
BATCH_SIZE = args.batch_size
SEQ_LEN = args.seq_len
FRAME_SIZE = args.frame_size
LR = args.lr
DATA_PATH = args.data_dir_path
CLASSES = args.class_list
NUM_WORKERS = args.num_workers
CALLBACKS = args.callbacks

# Error handling
if not CLASSES == None:
    # Checking valid class list
    ucf_class_list = data_setup.Dataset(data_path=DATA_PATH).classes
    for i in CLASSES:
        if i not in ucf_class_list:
            print(f'[ERROR] "{i}" is a wrong class name.')
            print(f'[INFO] Kindly select classes from this list: {ucf_class_list}')
            exit()
    
    # Checking total classes used for training
    if not len(CLASSES) >= 3:
        print(f'[ERROR] The Class list "{CLASSES}", contains less classes than the requirement.')
        print('[INFO] Minimum required classes in class list is 3. If you want to use the whole dataset than do not use this flag.')
        exit()
    
print(f'\n[INFO] Training a {MODEL_NAME} model for {NUM_EPOCHS} epochs with batch size {BATCH_SIZE} and a learning rate of {LR}')

# Creating dataset using data_setup script
dataset = data_setup.Dataset(data_path=DATA_PATH, 
                             seq_len=SEQ_LEN,
                             frame_size=FRAME_SIZE,
                             batch_size=BATCH_SIZE,
                             class_list=CLASSES)
train_ds, test_ds = dataset.dataset_pipeline()

# Creating model using the model_builder script
# Getting number of classes
if CLASSES == None:
    NUM_CLASSES = 50
else:
    NUM_CLASSES = len(CLASSES)

# Selecting a model and creating it.
if MODEL_NAME == 'ConvLSTM':
    model = model_builder.CreateConvlstmModel(input_shape = (SEQ_LEN, FRAME_SIZE, FRAME_SIZE, 3), 
                                              num_classes = NUM_CLASSES)
    model.build((BATCH_SIZE, SEQ_LEN, FRAME_SIZE, FRAME_SIZE, 3))
    print(f'[INFO] Model "{MODEL_NAME}" is been constructed.')
elif MODEL_NAME == 'LRCN':
    model = model_builder.CreateLRCNModel(input_shape = (SEQ_LEN, FRAME_SIZE, FRAME_SIZE, 3), 
                                          num_classes = NUM_CLASSES)
    model.build((BATCH_SIZE, SEQ_LEN, FRAME_SIZE, FRAME_SIZE, 3))
    print(f'[INFO] Model "{MODEL_NAME}" is been constructed.')
    
# Compiling the model
model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
              metrics=['accuracy'])

# Setting up callbacks
if CALLBACKS == 'True':
    callbacks = [utils.tensorboard_callback(dir_name='training_logs', model_name=MODEL_NAME, exp_name=EXP_NAME),
                 utils.early_stopping_callback,
                 utils.reduce_lr_callback]
elif CALLBACKS == 'False':
    callbacks = [utils.tensorboard_callback(dir_name='training_logs', model_name=MODEL_NAME, exp_name=EXP_NAME)]

# Fitting the model
model.fit(train_ds,
          epochs=NUM_EPOCHS,
          steps_per_epoch=len(train_ds),
          validation_data=test_ds,
          validation_steps=len(test_ds),
          callbacks=callbacks,
          workers=NUM_WORKERS)

# Save the model
utils.save_model(model=model, dir_name='saved_model', model_name=MODEL_NAME, exp_name=EXP_NAME)
