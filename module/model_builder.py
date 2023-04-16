from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import ConvLSTM2D, Conv2D, MaxPooling3D, MaxPooling2D, LSTM, TimeDistributed, Dropout, Flatten, Dense

class CreateConvlstmModel(Model):
    """
    Constructs and Initiates a ConvLSTM model for video classification.

    Parameters: 
        input_shape: tuple, Input shape of the array that is feeded in the model.
                     Format of the input_shape should be (timesteps, height, width, channels)
        num_classes: int, Total number of classes that model needs to predict.

    Returns: Fully Constructed ConvLSTM Model.
    """
    def __init__(self, input_shape: tuple, num_classes: int):
        super(CreateConvlstmModel, self).__init__()
        self.input_block1 = Sequential([
            ConvLSTM2D(4, 3, activation='tanh', data_format='channels_last', recurrent_dropout=0.2, return_sequences=True, input_shape=input_shape),
            MaxPooling3D((1, 2, 2), padding='same', data_format='channels_last'),
            TimeDistributed(Dropout(0.2))
        ])
        self.block2 = Sequential([
            ConvLSTM2D(8, 3, activation='tanh', data_format='channels_last', recurrent_dropout=0.2, return_sequences=True),
            MaxPooling3D((1, 2, 2), padding='same', data_format='channels_last'),
            TimeDistributed(Dropout(0.2))
        ])
        self.block3 = Sequential([
            ConvLSTM2D(12, 3, activation='tanh', data_format='channels_last', recurrent_dropout=0.2, return_sequences=True),
            MaxPooling3D((1, 2, 2), padding='same', data_format='channels_last'),
            TimeDistributed(Dropout(0.2))
        ])
        self.block4 = Sequential([
            ConvLSTM2D(16, 3, activation='tanh', data_format='channels_last', recurrent_dropout=0.2, return_sequences=True),
            MaxPooling3D((1, 2, 2), padding='same', data_format='channels_last')
        ])
        self.classifier_block = Sequential([
            Flatten(),
            Dense(num_classes, activation='softmax')
        ])
    
    def call(self, x):
        x = self.input_block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return self.classifier_block(x)

class CreateLRCNModel(Model):
    """
    Constructs and Initiates a LRCN model for video classification.

    Parameters: 
        input_shape: tuple, Input shape of the array that is feeded in the model.
                     Format of the input_shape should be (timesteps, height, width, channels)
        num_classes: int, Total number of classes that model needs to predict.

    Returns: Fully Constructed LRCN Model.
    """
    def __init__(self, input_shape: tuple, num_classes: int):
        super(CreateLRCNModel, self).__init__()
        self.input_block1 = Sequential([
            TimeDistributed(Conv2D(16, 3, padding='same', activation='relu'), input_shape=input_shape),
            TimeDistributed(MaxPooling2D(4)),
            TimeDistributed(Dropout(0.25))
        ])
        self.block2 = Sequential([
            TimeDistributed(Conv2D(32, 3, padding='same', activation='relu')),
            TimeDistributed(MaxPooling2D(4)),
            TimeDistributed(Dropout(0.25))
        ])
        self.block3 = Sequential([
            TimeDistributed(Conv2D(64, 3, padding='same', activation='relu')),
            TimeDistributed(MaxPooling2D(2)),
            TimeDistributed(Dropout(0.25))
        ])
        self.block4 = Sequential([
            TimeDistributed(Conv2D(64, 3, padding='same', activation='relu')),
            TimeDistributed(MaxPooling2D(2))
        ])
        self.classifier_block = Sequential([
            TimeDistributed(Flatten()),
            LSTM(32),
            Dense(num_classes, activation='softmax')
        ])
        
    def call(self, x):
        x = self.input_block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return self.classifier_block(x)
