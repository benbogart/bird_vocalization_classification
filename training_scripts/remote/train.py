
import argparse
import os
import numpy as np
import pickle
import sys

from azureml.core import Run
from azureml.core import Workspace, Dataset

from DataGenerator import DataGenerator, AugDataGenerator
from LogToAzure import LogToAzure
from tensorflow import keras as K
from tensorflow.distribute import MirroredStrategy

from sklearn.preprocessing import LabelEncoder

from models import MODELS

import json

import subprocess

def process_arguments():

    # parse the parameters passed to the this script
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str,
                        dest='data_path',
                        default='../../data/audio_10sec', #local path for testing
                        help='data folder mounting point')

    parser.add_argument('--test-data-path', type=str,
                        dest='test_data_path',
                        default='../../data/audio_10sec', #local path for testing
                        help='data folder mounting point')

    parser.add_argument('--sr', type=int,
                        dest='sr',
                        default=22050,
                        help='sample rate of audio files')

    parser.add_argument('--offline', dest='online', action='store_const',
                        const=False, default=True,
                        help='Do not perform online (Azure specific) tasks')

    parser.add_argument('--model-name', type=str,
                         dest='model_name',
                         default='cnn1_audin_nmel_1',
                         help='name of model to build')

    parser.add_argument('--data-subset',
                        type=str,
                        dest='data_subset',
                        default='kaggle',
                        help='the subset of the data to use [all, kaggle].')

    parser.add_argument('--augment-position',
                        action='store_const',
                        dest='augment_position',
                        const=True, default=False,
                        help='Whether to choose clip position randomly')

    parser.add_argument('--augment-pitch',
                        action='store_const',
                        dest='augment_pitch',
                        const=True, default=False,
                        help='Whether to use pitch augmentation.')

    parser.add_argument('--augment-stretch',
                        action='store_const',
                        dest='augment_stretch',
                        const=True, default=False,
                        help='Wether to use time stretch augmentation')

    parser.add_argument('--epochs',
                        type=int,
                        dest='epochs',
                        default=5,
                        help='number of epochs to try.')

    parser.add_argument('--learning-rate',
                        type=float,
                        dest='learning_rate',
                        default=0.001,
                        help='learning rate.')

    print('Parsing Args...')
    args = parser.parse_args()
    return args

args = process_arguments()

# set seed for reproducibility
np.random.seed(867)

# output is written to log file, separate output from previous log entries.
print('-'*100)

args = process_arguments()
print('ARGS\n',args)

sr = args.sr
dt = 10 # to second

# if the run is online start logging
if args.online:
    run = Run.get_context()
    run.tag('model_name', args.model_name)
    run.tag('learning_rate', args.learning_rate)
    run.tag('data_subset', args.data_subset)

    print('Environment:',run.get_environment().name)
    runid = run.id

    logger_fname = f'outputs/{runid}-log_compute.csv'
    logger_pid = subprocess.Popen(
        ['python', 'log_gpu_cpu_stats.py',
         logger_fname,
         '--loop',  '0.2',  # Interval between measurements, in seconds (optional, default=1)
        ])
    print('Started logging compute utilisation, pid', logger_pid)

# for an offline run just set the run id to offline
else:
    runid = 'offline'


# get dataset name
if args.data_subset == 'kaggle_10sec_wav':
    label_file = 'data_kag_split_single_label.json'
    ext = '.wav'

elif args.data_subset == 'all_10sec_wav':
    label_file = 'data_split_single_label.json'
    ext = '.wav'

# elif args.data_subset == 'kaggle_full_length_wav':
#     label_file = 'data_kag_split_single_label.json'
#     # data_path = os.path.join(args.data_path, 'wav')
#     ext = '.wav'
#
# elif args.data_subset == 'all_full_length_wav':
#     label_file = 'data_split_single_label.json'
#     # data_path = os.path.join(args.data_path, 'wav')
#     ext = '.wav'

elif args.data_subset == 'kaggle_full_length_npy':
    label_file = 'data_kag_split_single_label.json'
    ext = '.npy'

elif args.data_subset == 'all_full_length_npy':
    label_file = 'data_split_single_label.json'
    ext = '.npy'

else:
    raise Exception(f'Invalid data_subset: {args.data_subset}')


with open(os.path.join('resources', label_file), 'r') as f:
    data =  json.load(f)

# get file lists
train_files = [os.path.join(args.data_path, name+ext)
               for name in data['train']['files']]
val_files = [os.path.join(args.test_data_path, name+'.wav')
               for name in data['val']['files']]
test_files = [os.path.join(args.test_data_path, name+'.wav')
               for name in data['test']['files']]

# print(val_files)

# get label lists
train_labels = np.array(data['train']['encoded_labels'])
val_labels = np.array(data['val']['encoded_labels'])
test_labels = np.array(data['test']['encoded_labels'])

# print number of files in each split to log
print('Num Train Files:', len(train_files))
print('Num Val Files:', len(val_files))
print('Num Test Files:', len(test_files))

# print number of files in each split to log
print('Num Train Labels:', len(train_labels))
print('Num Val Labels:', len(val_labels))
print('Num Test Labels:', len(test_labels))

classes = data['mapping']
n_classes = len(classes)

# Parallelize for multiple gpus
strategy = MirroredStrategy()

# get number of gpus (replicas) for batch_size calculation
n_gpus = strategy.num_replicas_in_sync
print('Running ', n_gpus, 'replicas in sync')

# set an azure tag for n_gpus
if args.online:
    run.tag('gpus', n_gpus)

# Create generators for importing the audio
BATCH_SIZE = 32


if args.augment_position or args.augment_pitch or args.augment_stretch:
    print('Creating train AugDataGenerator wtih pitch', args.augment_pitch,
          'stretch', args.augment_stretch)
    train_generator = AugDataGenerator(wav_paths=train_files[:32],
                                       labels=train_labels[:32],
                                       sr=sr,
                                       dt=dt,
                                       n_classes=len(classes),
                                       # audio_segment=args.augment_position,
                                       pitch_shift=args.augment_pitch,
                                       time_stretch=args.augment_stretch,
                                       multithread=args.multithread,
                                       batch_size=BATCH_SIZE*n_gpus)
    # print('Creating validation DataGenerator...')
    # val_generator = AugDataGenerator(wav_paths=val_files,
    #                                    labels=val_labels,
    #                                    sr=sr,
    #                                    dt=dt,
    #                                    n_classes=len(classes),
    #                                    audio_segment='start',
    #                                    pitch_shift=False,
    #                                    time_stretch=False,
    #                                    batch_size=BATCH_SIZE*n_gpus)
    #
    #
    # print('Creating test DataGenerator...')
    # test_generator = AugDataGenerator(wav_paths=test_files,
    #                                    labels=test_labels,
    #                                    sr=sr,
    #                                    dt=dt,
    #                                    n_classes=len(classes),
    #                                    audio_segment='start',
    #                                    pitch_shift=False,
    #                                    time_stretch=False,
    #                                    batch_size=BATCH_SIZE*n_gpus)
else:
    print('Creating train DataGenerator')
    train_generator = DataGenerator(wav_paths=train_files,
                                    labels=train_labels,
                                    sr=sr,
                                    dt=dt,
                                    n_classes=len(classes),
                                    batch_size=BATCH_SIZE*n_gpus)

print('Creating validation DataGenerator...')
val_generator = DataGenerator(wav_paths=val_files, #[:32],
                                labels=val_labels, #[:32],
                                sr=sr,
                                dt=dt,
                                n_classes=len(classes),
                                batch_size=BATCH_SIZE*n_gpus)


print('Creating test DataGenerator...')
test_generator = DataGenerator(wav_paths=test_files, #[:32],
                                labels=test_labels, #[:32],
                                sr=sr,
                                dt=dt,
                                n_classes=len(classes),
                                batch_size=BATCH_SIZE*n_gpus,
                                shuffle=False)


# variables in this block are parrallelized
with strategy.scope():

    # metrics
    metrics = [
        K.metrics.CategoricalAccuracy(),
        K.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy'),
        K.metrics.TopKCategoricalAccuracy(k=5, name='top_5_accuracy')
    ]

    # Construct model
    model = MODELS[args.model_name]
    model = model()

    model.compile(optimizer=K.optimizers.Adam(learning_rate=args.learning_rate),
                  loss='categorical_crossentropy',
                  metrics=metrics)

model.summary()

# callbacks
r_lr = K.callbacks.ReduceLROnPlateau(patience=5, factor=0.2)
cb = K.callbacks.EarlyStopping(patience=10)
mc = K.callbacks.ModelCheckpoint(filepath=f'outputs/{args.model_name}-{runid}.h5',
                                 save_best_only=True,
                                 save_weights_only=True)
tb = K.callbacks.TensorBoard(log_dir=f'logs/{args.model_name}/',
                          histogram_freq=1,
                          profile_batch=0)

callbacks = [r_lr, cb, mc, tb]

if args.online:
    callbacks.append(LogToAzure(run))

# fit model and store history
history = model.fit(train_generator,
                    steps_per_epoch=1, # only for quick testing
                    validation_data=val_generator,
                    epochs=args.epochs,
                    callbacks=callbacks)

print('Saving model history...')
os.makedirs('outputs', exist_ok = True)
with open(f'outputs/{args.model_name}-{runid}.history', 'wb') as f:
    pickle.dump(history.history, f)


print('Loading best model for testing...')
model.load_weights(f'outputs/{args.model_name}-{runid}.h5')

print('evaluating model on test set...')
model_val = model.evaluate(test_generator)

print('model_val len',len(model_val))
print('metrics len',len(metrics))

test_metrics = {}
for i, m in enumerate(metrics):
    print(f'test_{m.name}: {model_val[i+1]}')
    test_metrics['test_'+m.name] = model_val[i+1]
    if args.online:
        print('logging metrics...')
        run.log('test_'+m.name, np.float(model_val[i+1]))

print('Saving test metrics...')
os.makedirs('outputs', exist_ok=True)
with open(f'outputs/{args.model_name}-{runid}-test_metrics.plk', 'wb') as f:
    pickle.dump(test_metrics, f)

print('generating predictions on test set...')
test_pred = model.predict(test_generator)

print('saving test predictions...')
with open(f'outputs/{args.model_name}-{runid}-test_predictions.plk', 'wb') as f:
    pickle.dump(test_pred, f)

print('Done!')
print('-'*100)