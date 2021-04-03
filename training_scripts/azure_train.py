
import argparse
import os
import shutil
import numpy as np
import pickle

from azureml.core import Run, Workspace, Dataset, Experiment, ScriptRunConfig

from azureml.core.environment import Environment

# from tensorflow.keras.preprocessing.image import (ImageDataGenerator, array_to_img,
#                                        img_to_array) #, load_img)

#from AudioDataGenerator import AudioDataGenerator

#from tensorflow.keras import callbacks
#from tensorflow.keras import models
#from tensorflow.keras import layers
#from tensorflow.keras import losses

#from tensorflow.keras.applications.vgg16 import VGG16


# set seed for reproducibility
np.random.seed(867)

# output will be logged, separate output from previous log entries.
print('-'*100)

# def process_arguments(args):
#
#     # parse the parameters passed to the this script
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--data-path', type=str,
#                         dest='data_path',
#                         default='data/audio_split',
#                         help='data folder mounting point')
#
#     parser.add_argument('--model-name', type=str,
#                         dest='model_name',
#                         default='baseline',
#                         help='name of model to build')
#     parser.add_argument('--optimizer',
#                         type=str,
#                         dest='optimizer',
#                         default='sgd',
#                         help='optimizer to use')
#     parser.add_argument('--epochs',
#                         type=int,
#                         dest='epochs',
#                         default=5,
#                         help='number of epochs to try.')
#     args = parser.parse_args()
#     return argparse


# Load the stored workspace
ws = Workspace.from_config()

# Get the registered dataset from azure
dataset = Dataset.get_by_name(ws, name='birdsongs')

## Try with our saved image
env = Environment.get(workspace=ws, name="birdsong-env-gpu")

# set the expiriment
experiment_name = 'test'
exp = Experiment(workspace=ws, name=experiment_name)

# get our compoute cluster
# for cnn we will use a gpu cluster
compute_name = "gpu-cluster-NC6"
compute_target = ws.compute_targets[compute_name]

args = ['--data-path', dataset.as_named_input('input').as_mount(),
        '--epochs', 50]

script_path = 'azure/test'

src = ScriptRunConfig(source_directory=script_path,
                      script='train.py',
                      arguments=args,
                      compute_target=compute_target,
                      environment=env)

run = exp.submit(config=src)


# Add name and tags for tracking
run.add_properties({'name': 'baseline'})
# runs.tag('class', 'Xception')
# runs.tag('optimizer', 'sgd')
print(run.id)
