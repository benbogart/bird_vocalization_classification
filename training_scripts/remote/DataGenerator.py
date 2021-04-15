from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import numpy as np
from scipy.io import wavfile
import librosa
import multiprocessing
from joblib import Parallel, delayed
import time

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, wav_paths, labels, sr, dt, n_classes,
                 batch_size=32, shuffle=True):
        self.wav_paths = wav_paths
        self.labels = labels
        self.sr = sr
        self.dt = dt
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.shuffle = True
        self.on_epoch_end()


    def __len__(self):
        return int(np.floor(len(self.wav_paths) / self.batch_size))


    def __getitem__(self, index):
        # start_time = time.time()
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        wav_paths = [self.wav_paths[k] for k in indexes]
        labels = [self.labels[k] for k in indexes]

        # generate a batch of time data
        X = np.empty((self.batch_size, int(self.sr*self.dt), 1), dtype=np.float32)

        Y = np.empty((self.batch_size, self.n_classes), dtype=np.float32)

        for i, (path, label) in enumerate(zip(wav_paths, labels)):
            # rate, wav = wavfile.read(path)
            wav, rate = librosa.load(path, sr=self.sr, duration=self.dt)
            X[i,] = wav.reshape(-1, 1)
            Y[i,] = to_categorical(label, num_classes=self.n_classes)
            #Y[i] = label

            #print('X.shape:', X.shape)
            #print('Y.shape:', Y.shape)
            # print('Seconds to get batch', time.time() - start_time)
        return X, Y


    def on_epoch_end(self):
        self.indexes = np.arange(len(self.wav_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)


class AugDataGenerator(tf.keras.utils.Sequence):
    '''Data Generator that augments audio data and loads directly from mp3
    files.  This Generator pulls random 10 second clips from mp3 files and
    performs basic pitch and time stretch augmentation on them'''

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
        self.shuffle = True
        self.accesstimes = []
        self.on_epoch_end()

        self.cpus = multiprocessing.cpu_count()
        print('num_cpus for audio processing:', self.cpus)

    def __len__(self):
        return int(np.floor(len(self.wav_paths) / self.batch_size))


    def __getitem__(self, index):
        start_time = time.time()
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        wav_paths = [self.wav_paths[k] for k in indexes]
        labels = [self.labels[k] for k in indexes]

        # generate a batch of time data
        # X = np.empty((self.batch_size, int(self.sr*self.dt), 1), dtype=np.float32)

        # generate X with joblib for parralezation

        if self.multithread:
            d = [delayed(self.augment_audio)(wav_path,
                                        pitch_shift=self.pitch_shift,
                                        stretch=self.time_stretch)
                                            for wav_path in wav_paths]
            X = Parallel(n_jobs=self.cpus, verbose=0)(d)
            X = np.array(X)

        else:
            X = np.empty((self.batch_size, int(self.sr*self.dt), 1), dtype=np.float32)
            for i, wav_path in enumerate(wav_paths):
                wav = self.augment_audio(wav_path,
                                         pitch_shift=self.pitch_shift,
                                         stretch=self.time_stretch)
                X[i,] = wav

        Y = np.empty((self.batch_size, self.n_classes), dtype=np.float32)

        for i, (path, label) in enumerate(zip(wav_paths, labels)):
            # rate, wav = wavfile.read(path)
            # wav, rate = librosa.load(path, sr=self.sr)
            # X[i,] = wav.reshape(-1, 1)

            # X[i,] = self.augment_audio(path,
            #                            pitch_shift=self.pitch_shift,
            #                            stretch=self.time_stretch)

            Y[i,] = to_categorical(label, num_classes=self.n_classes)
            #Y[i] = label

            #print('X.shape:', X.shape)
            #print('Y.shape:', Y.shape)
            self.accesstimes.append(time.time() - start_time)

        return X, Y


    def on_epoch_end(self):
        self.indexes = np.arange(len(self.wav_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)
        print('average file access time:', np.mean(self.accesstimes))
        self.accesstimes = []

    def get_random_slice(self, wav, samples):
        # for long files take a random slice
        # print('samples', samples)

        total_samples = len(wav)
        if total_samples > samples:
            start_idx = np.random.choice(list(range(total_samples - samples)))
            # print(start_idx)
            return wav[start_idx:start_idx+samples]
        # pad shorter files
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
        rng = np.random.default_rng()
        semi_tones = rng.normal(loc=0, scale=1)
        return librosa.effects.pitch_shift(wav, self.sr, semi_tones)

    def random_stretch(self, wav):
        rng = np.random.default_rng()
        stretch = rng.normal(loc=1, scale=0.1)
        return librosa.effects.time_stretch(wav, stretch)

    def augment_audio(self, path, pitch_shift = True, stretch=True):
        wav = np.load(path, mmap_mode='c')

        total_samples = wav.shape[0]
        target_samples = 22050*10

        # is the file longer than our target length
        long_file = total_samples > target_samples


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

        wav = wav.reshape(-1, 1)
        return wav
