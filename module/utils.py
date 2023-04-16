import os
import datetime as dt
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ReduceLROnPlateau

def tensorboard_callback(dir_name: str, model_name: str, exp_name: str):
    """
    Creates Tensorboard Callback
    Parameters: 
        dir_name: A string to save tensorboard data in a directory.
        model_name: A string for the model name.
        exp_name: A string for the experiment name.
    Returns: A pre-configured tensorboard callback.
    """
    # Creating a tensorboard callback
    log_dir = os.path.join(dir_name, model_name, exp_name, dt.datetime.now().strftime('%Y-%m-%d-%H:%M:%S'))
    tensorboard_callback = TensorBoard(log_dir=log_dir)
    print(f'[INFO] Saving Tensorboard log files to: {log_dir}')
    return tensorboard_callback
    
# Creating a early stopping callback
early_stopping_callback = EarlyStopping(monitor='val_loss',
                                        patience=20,
                                        mode='min',
                                        verbose=1,
                                        restore_best_weights=True)
    
# Creating a reduce learning rate callback
reduce_lr_callback = ReduceLROnPlateau(monitor='val_loss',
                                       mode='min',
                                       factor=0.2,
                                       patience=10,
                                       verbose=1,
                                       min_lr=1e-7)

def save_model(model, dir_name: str, model_name: str, exp_name: str):
    """
    A function to save a tensorflow model.
    Parameters: 
        model: A trained model.
        dir_name: A string to save whole model data in a directory.
        model_name: A string for the model name.
        exp_name: A string for the experiment name.
    """
    filepath = os.path.join(dir_name, model_name, exp_name, dt.datetime.now().strftime('%Y-%m-%d-%H:%M:%S'))
    model.save(filepath=filepath, save_format='tf')
    print(f'[INFO] "{model_name}" Model is been saved to directory: {filepath}')
