from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import numpy as np
from scipy.io import wavfile
import librosa
import multiprocessing
from joblib import Parallel, delayed
import time

class DataGenerator(tf.keras.utils.Sequence):
    '''Simple audio file data generator'''

    def __init__(self, wav_paths, labels, sr, dt, n_classes,
                 batch_size=32, shuffle=True):
        self.wav_paths = wav_paths
        self.labels = labels
        self.sr = sr
        self.dt = dt
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.accesstimes = np.array([0])
        self.on_epoch_end()

        # Print Id to log for verification
        print('Using DataGenerator')


    def __len__(self):
        # iterations per epoch
        return int(np.floor(len(self.wav_paths) / self.batch_size))


    def __getitem__(self, index):

        # store access time to log
        start_time = time.time()

        # get indexes for next batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # get paths and labels from indexes
        wav_paths = [self.wav_paths[k] for k in indexes]
        labels = [self.labels[k] for k in indexes]

        # generate a batch of data
        X = np.empty((self.batch_size, int(self.sr*self.dt), 1), dtype=np.float32)

        Y = np.empty((self.batch_size, self.n_classes), dtype=np.float32)

        for i, (path, label) in enumerate(zip(wav_paths, labels)):
            # load the actual audio file
            wav, rate = librosa.load(path, sr=self.sr, duration=self.dt)
            X[i,] = wav.reshape(-1, 1)
            Y[i,] = to_categorical(label, num_classes=self.n_classes)

            # store time elapsed
            self.accesstimes.append(time.time() - start_time)

        return X, Y


    def on_epoch_end(self):

        # reset index
        self.indexes = np.arange(len(self.wav_paths))

        # suffle index if shufle = True
        if self.shuffle:
            np.random.shuffle(self.indexes)

        # Print to log the average file access time
        print('average file access time:', np.mean(self.accesstimes))

        # Rest for next epoch
        self.accesstimes = []


class AugDataGenerator(tf.keras.utils.Sequence):
    '''Data Generator that augments audio data from npy files.  This Generator
    pulls random 10 second clips from mp3 files and performs basic pitch and
    time stretch augmentation on them'''

    def __init__(self, wav_paths, labels, sr, dt, n_classes,
                 batch_size=32, pitch_shift=True, time_stretch=True,
                 multithread=False, shuffle=True):
        self.wav_paths = wav_paths
        self.labels = labels
        self.sr = sr
        self.dt = dt
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.pitch_shift = pitch_shift
        self.time_stretch = time_stretch
        self.multithread = multithread
        self.shuffle = shuffle
        self.accesstimes = [0] # avoids wtih on_epoch_end at startup
        self.on_epoch_end()

        # Print to log
        print('Using AugDataGenerator')
        if self.pitch_shift:
            print('  - with pitch_shift')
        if self.time_stretch:
            print('  - with time_stretch')

        # get number of CPUs for multithreaded loading
        self.cpus = multiprocessing.cpu_count()
        print('num_cpus for audio processing:', self.cpus)

    def __len__(self):
        # number of batches in epoch
        return int(np.floor(len(self.wav_paths) / self.batch_size))


    def __getitem__(self, index):

        # set time for logging
        start_time = time.time()

        # Get indexes for next batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # get file paths and labels from indexes
        wav_paths = [self.wav_paths[k] for k in indexes]
        labels = [self.labels[k] for k in indexes]

        if self.multithread:
            # process files in parallel on all cpus
            d = [delayed(self.augment_audio)(wav_path,
                                        pitch_shift=self.pitch_shift,
                                        stretch=self.time_stretch)
                                            for wav_path in wav_paths]
            X = Parallel(n_jobs=self.cpus, verbose=0)(d)
            X = np.array(X)

        else:
            # process files sequentially
            X = np.empty((self.batch_size, int(self.sr*self.dt), 1), dtype=np.float32)
            for i, wav_path in enumerate(wav_paths):
                wav = self.augment_audio(wav_path,
                                         pitch_shift=self.pitch_shift,
                                         stretch=self.time_stretch)
                X[i,] = wav

        # set labels
        Y = np.empty((self.batch_size, self.n_classes), dtype=np.float32)

        for i, (path, label) in enumerate(zip(wav_paths, labels)):
            Y[i,] = to_categorical(label, num_classes=self.n_classes)

        # store accesstime
        self.accesstimes.append(time.time() - start_time)

        return X, Y


    def on_epoch_end(self):
        # reset indexes
        self.indexes = np.arange(len(self.wav_paths))

        # shuffle if shuffle == True
        if self.shuffle:
            np.random.shuffle(self.indexes)

        # Log accestime and reset
        print('average file access time:', np.mean(self.accesstimes))
        self.accesstimes = []

    def get_random_slice(self, wav, samples):
        '''Take a random 10 second slice from files > 10 sec'''

        total_samples = len(wav)
        if total_samples > samples:
            start_idx = np.random.choice(list(range(total_samples - samples)))
            return wav[start_idx:start_idx+samples]
        else:
            print('wav length must be longer than samples')

    def pad_audio(self, wav, samples):
        total_samples = len(wav)
        pad_length = samples - total_samples

        if total_samples > samples:
            print('wav is longer than requested samples.  Cannot pad.')
            return False

        # randomly divide padding between start and end of file
        front_pad = np.random.choice(range(pad_length))
        back_pad = samples - total_samples - front_pad

        return np.pad(wav, (front_pad, back_pad), constant_values=(0,0))

    def random_pitch_shift(self, wav):

        # Randomly generate pitch shift amount
        rng = np.random.default_rng()
        semi_tones = rng.normal(loc=0, scale=1)
        return librosa.effects.pitch_shift(wav, self.sr, semi_tones)

    def random_stretch(self, wav):

        # Randomly generate stretch amount
        rng = np.random.default_rng()
        stretch = rng.normal(loc=1, scale=0.1)
        return librosa.effects.time_stretch(wav, stretch)

    def augment_audio(self, path, pitch_shift = True, stretch=True):

        try:
            # NOTE: use of mmap_mode here corrupted files
            wav = np.load(path) #, mmap_mode='r')
        except:
            print('Could not open file:', path)
            raise Exception('Could not open file:', path)

        total_samples = wav.shape[0]
        target_samples = 22050*10

        if total_samples > target_samples:
            # cut a longfile down to no more than 1.5x length of sample
            wav = self.get_random_slice(wav, target_samples)

        if pitch_shift:
            wav = self.random_pitch_shift(wav)

        if stretch:
            wav = self.random_stretch(wav)

            # we have changed the length of the file so recalculate
            total_samples = wav.shape[0]

            # if it was made longer
            if total_samples > target_samples:
                wav = self.get_random_slice(wav, target_samples)

        if total_samples < target_samples:
            # pad short files
            wav = self.pad_audio(wav, target_samples)

        wav = wav.reshape(-1, 1) # for tensorflow input
        return wav
