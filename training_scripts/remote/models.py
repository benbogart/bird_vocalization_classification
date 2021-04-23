import tensorflow.keras as K
import kapre

def construct_cnn1_audin_nmel_1():
    'n_mels'
    sample_rate = 22050
    n_fft = 2048 # frame size
    hop_length = 512
    n_mels=128
    mel_f_min=0.0
    mel_f_max=None
    return_decibel=False

    model = _construct_cnn1_audin(sample_rate=sample_rate,
                                 n_fft=n_fft,
                                 n_mels=n_mels,
                                 mel_f_min=mel_f_min,
                                 mel_f_max=mel_f_max,
                                 return_decibel=return_decibel)
    return model

def construct_cnn1_audin_nmel_2():
    'n_mels'
    sample_rate = 22050
    n_fft = 2048 # frame size
    hop_length = 512
    n_mels=256
    mel_f_min=0.0
    mel_f_max=None
    return_decibel=False

    model = _construct_cnn1_audin(sample_rate=sample_rate,
                                 n_fft=n_fft,
                                 n_mels=n_mels,
                                 mel_f_min=mel_f_min,
                                 mel_f_max=mel_f_max,
                                 return_decibel=return_decibel)

    return model

def construct_cnn1_audin_nmel_dcbl_1():
    sample_rate = 22050
    n_fft = 2048
    hop_length = 512
    # n_hop = n_fft // 2
    n_mels=128
    mel_f_min=0.0
    mel_f_max=None
    return_decibel=True

    model = _construct_cnn1_audin(sample_rate=sample_rate,
                                 n_fft=n_fft,
                                 n_mels=n_mels,
                                 mel_f_min=mel_f_min,
                                 mel_f_max=mel_f_max,
                                 return_decibel=return_decibel)
    return model

def construct_cnn1_audin_nmel_dcbl_2():
    'n_mels'
    sample_rate = 22050
    n_fft = 2048 # frame size
    hop_length = 512
    n_mels=256
    mel_f_min=0.0
    mel_f_max=None
    return_decibel=True

    model = _construct_cnn1_audin(sample_rate=sample_rate,
                                 n_fft=n_fft,
                                 n_mels=n_mels,
                                 mel_f_min=mel_f_min,
                                 mel_f_max=mel_f_max,
                                 return_decibel=return_decibel)
    return model

    model = _construct_cnn1_audin(sample_rate=sample_rate,
                                 n_fft=n_fft,
                                 n_mels=n_mels,
                                 mel_f_min=mel_f_min,
                                 mel_f_max=mel_f_max,
                                 return_decibel=return_decibel)
    return model

def construct_cnn1_audin_nffthl_1():
    'hop_length'
    sample_rate = 22050
    n_fft = 2048 # frame size
    hop_length = 256
    n_mels=128
    mel_f_min=0.0
    mel_f_max=None
    return_decibel=False

    model = _construct_cnn1_audin(sample_rate=sample_rate,
                                 n_fft=n_fft,
                                 n_mels=n_mels,
                                 mel_f_min=mel_f_min,
                                 mel_f_max=mel_f_max,
                                 return_decibel=return_decibel)
    return model

def construct_cnn1_audin_nffthl_2():
    'hop_legnth + n_fft'
    sample_rate = 22050
    n_fft = 1024 # frame size
    hop_length = 256
    n_mels=128
    mel_f_min=0.0
    mel_f_max=None
    return_decibel=False

    model = _construct_cnn1_audin(sample_rate=sample_rate,
                                 n_fft=n_fft,
                                 n_mels=n_mels,
                                 mel_f_min=mel_f_min,
                                 mel_f_max=mel_f_max,
                                 return_decibel=return_decibel)
    return model

def construct_cnn1_audin_freq_1():
    'frequency'
    sample_rate = 22050
    n_fft = 2048 # frame size
    hop_length = 512
    n_mels=128
    mel_f_min=500
    mel_f_max=None
    return_decibel=False

    model = _construct_cnn1_audin(sample_rate=sample_rate,
                                 n_fft=n_fft,
                                 n_mels=n_mels,
                                 mel_f_min=mel_f_min,
                                 mel_f_max=mel_f_max,
                                 return_decibel=return_decibel)

    return model

def construct_cnn1_audin_freq_2():
    'frequency'
    sample_rate = 22050
    n_fft = 2048 # frame size
    hop_length = 512
    n_mels=128
    mel_f_min=1000
    mel_f_max=None
    return_decibel=False

    model = _construct_cnn1_audin(sample_rate=sample_rate,
                                 n_fft=n_fft,
                                 n_mels=n_mels,
                                 mel_f_min=mel_f_min,
                                 mel_f_max=mel_f_max,
                                 return_decibel=return_decibel)

    return model

def _construct_cnn1_audin(sample_rate, n_fft, n_mels, mel_f_min, mel_f_max, return_decibel):
    '''Simple CNN with 3 CNN Layers, 2 dense layers.
    Accepts 10 second audio as input.'''


    input_shape = (sample_rate * 10, 1) # mono 10 seconds at 22050hz

    model = K.Sequential()
    composed_melgram_layer = \
        kapre.composed.get_melspectrogram_layer(input_shape=input_shape,
                                                sample_rate=sample_rate,
                                                n_fft=n_fft,
                                                n_mels=n_mels,
                                                mel_f_min=mel_f_min,
                                                mel_f_max=mel_f_max,
                                                return_decibel=return_decibel)

    # decompose the layers the model can be saved
    for layer in composed_melgram_layer.layers:
        model.add(layer)

    model.add(K.layers.BatchNormalization())
    model.add(K.layers.Conv2D(32, (3,3),
                              padding='same',
                              activation='relu',
                              kernel_initializer='he_normal'))
    model.add(K.layers.MaxPooling2D((2,2), strides=None, padding='valid'))

    model.add(K.layers.BatchNormalization())
    model.add(K.layers.Conv2D(64, (3,3),
                              padding='same',
                              activation='relu',
                              kernel_initializer='he_normal'))
    model.add(K.layers.MaxPooling2D((2,2), strides=None, padding='valid'))

    model.add(K.layers.BatchNormalization())
    model.add(K.layers.Conv2D(128, (3,3),
                              padding='same',
                              activation='relu',
                              kernel_initializer='he_normal'))
    model.add(K.layers.MaxPooling2D((3,3), strides=None, padding='valid'))

    model.add(K.layers.GlobalMaxPooling2D())

    model.add(K.layers.Dense(1028, activation='relu'))
    model.add(K.layers.Dense(512, activation='relu'))
    model.add(K.layers.Dense(264, activation='softmax'))

    return model

def construct_cnn1_audin_l2reg_1():
    'Best model abouve (construct_cnn1_audin_nmel_1) with l2 regularization'
    model = _construct_cnn1_audin_l2reg(lam = 0.0001)
    return model

def construct_cnn1_audin_l2reg_2():
    'Best model abouve (construct_cnn1_audin_nmel_1) with l2 regularization'
    return _construct_cnn1_audin_l2reg(lam = 0.001)

def construct_cnn1_audin_l2reg_3():
    'Best model abouve (construct_cnn1_audin_nmel_1) with l2 regularization'
    return _construct_cnn1_audin_l2reg(lam = 0.01)

def construct_cnn1_audin_l2reg_4():
    'Best model abouve (construct_cnn1_audin_nmel_1) with l2 regularization'
    return _construct_cnn1_audin_l2reg(lam = 0.1)

def _construct_cnn1_audin_l2reg(lam = .0001):
    '''Add in regularization for overfitting'''

    sample_rate = 22050
    n_fft = 2048 # frame size
    hop_length = 512
    n_mels=128
    mel_f_min=0.0
    mel_f_max=None
    return_decibel=False


    input_shape = (sample_rate * 10, 1) # mono 10 seconds at 22050hz

    model = K.Sequential()
    composed_melgram_layer = \
        kapre.composed.get_melspectrogram_layer(input_shape=input_shape,
                                                sample_rate=sample_rate,
                                                n_fft=n_fft,
                                                n_mels=n_mels,
                                                mel_f_min=mel_f_min,
                                                mel_f_max=mel_f_max,
                                                return_decibel=return_decibel)

    # decompose the layers the model can be saved
    for layer in composed_melgram_layer.layers:
        model.add(layer)

    model.add(K.layers.BatchNormalization())
    model.add(K.layers.Conv2D(32, (3,3),
                              padding='same',
                              activation='relu',
                              kernel_initializer='he_normal'))
    model.add(K.layers.MaxPooling2D((2,2), strides=None, padding='valid'))

    model.add(K.layers.BatchNormalization())
    model.add(K.layers.Conv2D(64, (3,3),
                              padding='same',
                              activation='relu',
                              kernel_initializer='he_normal'))
    model.add(K.layers.MaxPooling2D((2,2), strides=None, padding='valid'))

    model.add(K.layers.BatchNormalization())
    model.add(K.layers.Conv2D(128, (3,3),
                              padding='same',
                              activation='relu',
                              kernel_initializer='he_normal'))
    model.add(K.layers.MaxPooling2D((3,3), strides=None, padding='valid'))

    model.add(K.layers.GlobalMaxPooling2D())


    model.add(K.layers.Dense(1028, activation='relu',
                             kernel_regularizer = K.regularizers.l2(lam)))
    model.add(K.layers.Dense(512, activation='relu',
                             kernel_regularizer = K.regularizers.l2(lam)))
    model.add(K.layers.Dense(264, activation='softmax'))

    return model

def construct_cnn1_audin_drp_1():
    return _construct_cnn1_audin_drp(rate = 0.2)

def construct_cnn1_audin_drp_2():
    return _construct_cnn1_audin_drp(rate = 0.3)

def construct_cnn1_audin_drp_3():
    return _construct_cnn1_audin_drp(rate = 0.4)

def construct_cnn1_audin_drp_4():
    return _construct_cnn1_audin_drp(rate = 0.5)

def _construct_cnn1_audin_drp(rate = 0.3):
    '''Add Dropout for overfitting'''

    sample_rate = 22050
    n_fft = 2048 # frame size
    hop_length = 512
    n_mels=128
    mel_f_min=0.0
    mel_f_max=None
    return_decibel=False


    input_shape = (sample_rate * 10, 1) # mono 10 seconds at 22050hz

    model = K.Sequential()
    composed_melgram_layer = \
        kapre.composed.get_melspectrogram_layer(input_shape=input_shape,
                                                sample_rate=sample_rate,
                                                n_fft=n_fft,
                                                n_mels=n_mels,
                                                mel_f_min=mel_f_min,
                                                mel_f_max=mel_f_max,
                                                return_decibel=return_decibel)

    # decompose the layers the model can be saved
    for layer in composed_melgram_layer.layers:
        model.add(layer)

    model.add(K.layers.BatchNormalization())
    model.add(K.layers.Conv2D(32, (3,3),
                              padding='same',
                              activation='relu',
                              kernel_initializer='he_normal'))
    model.add(K.layers.MaxPooling2D((2,2), strides=None, padding='valid'))

    model.add(K.layers.BatchNormalization())
    model.add(K.layers.Conv2D(64, (3,3),
                              padding='same',
                              activation='relu',
                              kernel_initializer='he_normal'))
    model.add(K.layers.MaxPooling2D((2,2), strides=None, padding='valid'))

    model.add(K.layers.BatchNormalization())
    model.add(K.layers.Conv2D(128, (3,3),
                              padding='same',
                              activation='relu',
                              kernel_initializer='he_normal'))
    model.add(K.layers.MaxPooling2D((3,3), strides=None, padding='valid'))

    model.add(K.layers.GlobalMaxPooling2D())


    model.add(K.layers.Dense(1028, activation='relu'))
    model.add(K.layers.Dropout(rate))

    model.add(K.layers.Dense(512, activation='relu'))
    model.add(K.layers.Dropout(rate))

    model.add(K.layers.Dense(264, activation='softmax'))

    return model

def construct_milsed_2block_dense():
    return _construct_milsed_block(2)

def construct_milsed_3block_dense():
    return _construct_milsed_block(3)

def construct_milsed_4block_dense():
    return _construct_milsed_block(4)

def construct_milsed_5block_dense():
    return _construct_milsed_block(5)

def construct_milsed_6block_dense():
    return _construct_milsed_block(6)

def construct_milsed_7block_dense():
    return _construct_milsed_block(7)

def construct_milsed_8block_dense():
    return _construct_milsed_block(8)

def construct_milsed_7block_dense_drp_1():
    return _construct_milsed_block(5, 0.2)

def construct_milsed_7block_dense_drp_2():
    return _construct_milsed_block(5, 0.3)

def construct_milsed_7block_dense_drp_3():
    return _construct_milsed_block(5, 0.4)

def construct_milsed_7block_dense_drp_4():
    return _construct_milsed_block(5, 0.5)

def _construct_milsed_block(num_blocks, dropout_rate = False):
    '''Variable Convolution Block model inspired by milsed'''

    sample_rate = 22050
    input_shape = (sample_rate * 10, 1) # mono 10 seconds at 22050hz
    'n_mels'
    n_fft = 2048 # frame size
    hop_length = 256
    n_mels=256
    mel_f_min=0.0
    mel_f_max=None
    return_decibel=True
    model = K.Sequential()
    composed_melgram_layer = \
        kapre.composed.get_melspectrogram_layer(input_shape=input_shape,
                                                sample_rate=sample_rate,
                                                n_fft=n_fft,
                                                n_mels=n_mels,
                                                mel_f_min=mel_f_min,
                                                mel_f_max=mel_f_max,
                                                return_decibel=return_decibel)

    # decompose the layers the model can be saved
    for layer in composed_melgram_layer.layers:
        model.add(layer)

    model.add(K.layers.BatchNormalization())

    # add blocks
    n_filters = 16
    for block in range(num_blocks):
        model.add(K.layers.Convolution2D(n_filters, (3, 3),
                                       padding='same',
                                       activation='relu',
                                       kernel_initializer='he_normal'))
        model.add(K.layers.BatchNormalization())
        model.add(K.layers.Convolution2D(n_filters, (3, 3),
                                       padding='same',
                                       activation='relu',
                                       kernel_initializer='he_normal'))
        model.add(K.layers.BatchNormalization())
        model.add(K.layers.MaxPooling2D((2,2), padding='valid'))

        # double the number of filters for the next block
        n_filters *= 2

    model.add(K.layers.GlobalMaxPooling2D())

    model.add(K.layers.Dense(1028, activation='relu'))
    if dropout_rate:
        model.add(K.layers.Dropout(dropout_rate))

    model.add(K.layers.Dense(512, activation='relu'))
    if dropout_rate:
        model.add(K.layers.Dropout(dropout_rate))

    model.add(K.layers.Dense(264, activation='softmax'))

    return model


MODELS={
    'cnn1_audin_nmel_1':construct_cnn1_audin_nmel_1,
    'cnn1_audin_nmel_2':construct_cnn1_audin_nmel_2,
    'cnn1_audin_nmel_dcbl_1':construct_cnn1_audin_nmel_dcbl_1,
    'cnn1_audin_nmel_dcbl_2':construct_cnn1_audin_nmel_dcbl_2,
    'cnn1_audin_nffthl_1':construct_cnn1_audin_nffthl_1,
    'cnn1_audin_nffthl_2':construct_cnn1_audin_nffthl_2,
    'cnn1_audin_freq_1':construct_cnn1_audin_freq_1,
    'cnn1_audin_freq_2':construct_cnn1_audin_freq_2,
    'cnn1_audin_l2reg_1': construct_cnn1_audin_l2reg_1,
    'cnn1_audin_l2reg_2': construct_cnn1_audin_l2reg_2,
    'cnn1_audin_l2reg_3': construct_cnn1_audin_l2reg_3,
    'cnn1_audin_l2reg_4': construct_cnn1_audin_l2reg_4,
    'cnn1_audin_drp_1':construct_cnn1_audin_drp_1,
    'cnn1_audin_drp_2':construct_cnn1_audin_drp_2,
    'cnn1_audin_drp_3':construct_cnn1_audin_drp_3,
    'cnn1_audin_drp_4':construct_cnn1_audin_drp_4,
    'milsed_2block_dense':construct_milsed_2block_dense,
    'milsed_3block_dense':construct_milsed_3block_dense,
    'milsed_4block_dense':construct_milsed_4block_dense,
    'milsed_5block_dense':construct_milsed_5block_dense,
    'milsed_6block_dense':construct_milsed_6block_dense,
    'milsed_7block_dense':construct_milsed_7block_dense,
    'milsed_8block_dense':construct_milsed_8block_dense,
    'milsed_7block_dense_drp_1':construct_milsed_7block_dense_drp_1,
    'milsed_7block_dense_drp_2':construct_milsed_7block_dense_drp_2,
    'milsed_7block_dense_drp_3':construct_milsed_7block_dense_drp_3,
    'milsed_7block_dense_drp_4':construct_milsed_7block_dense_drp_4
    }
