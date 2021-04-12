
import argparse
import os
import numpy as np
import csv
from datetime import datetime
from azureml.core import Run, Workspace, Dataset, Experiment, ScriptRunConfig
from azureml.core.environment import Environment

# set seed for reproducibility
np.random.seed(867)

def process_arguments():
    # parse the parameters passed to the this script
    parser = argparse.ArgumentParser()

    parser.add_argument('--model-name', type=str,
                        dest='model_name',
                        required=True,
                        help='name of model to build')
    parser.add_argument('--epochs',
                        type=int,
                        dest='epochs',
                        default=100,
                        help='number of epochs to try.')
    parser.add_argument('--gpus',
                        type=int,
                        dest='gpus',
                        default=1,
                        help='number of gpus to use for training.')
    parser.add_argument('--data_subset',
                        type=str,
                        dest='data_subset',
                        choices=['all', 'kaggle'],
                        default='kaggle',
                        help='the subset of the data to use [all, kaggle].')
    parser.add_argument('--test', dest='test', action='store_const',
                        const=True, default=False,
                        help='Use test expiriment')
    args = parser.parse_args()
    return args

args = process_arguments()
print(args)

# Load the stored workspace
ws = Workspace.from_config()

# Get the registered dataset from azure
dataset = Dataset.get_by_name(ws, name='birdsongs')

## Try with our saved image
env = Environment.get(workspace=ws, name="birdsong-env-gpu")

# set the expiriment name
if args.test:
    experiment_name = 'test'
else:
    experiment_name = 'birdsongs_2'
exp = Experiment(workspace=ws, name=experiment_name)

# get the compute cluster
if args.gpus == 1:
    compute_name = "gpu-cluster-NC6"
elif args.gpus == 2:
    compute_name = "gpu-cluster-NC12"
elif args.gpus == 4:
    compute_name = "gpu-cluster-NC24"
else:
    raise Exception(f'{args.gpus} is an invalid value for gpus')

compute_target = ws.compute_targets[compute_name]

# set the args to pass to the training script on azure
azure_args = ['--data-path', dataset.as_named_input('input').as_mount(),
              '--model-name', args.model_name,
              '--epochs', args.epochs,
              '--data-subset', args.data_subset]

script_path = 'remote'

# setup the run details
src = ScriptRunConfig(source_directory=script_path,
                      script='train.py',
                      arguments=azure_args,
                      compute_target=compute_target,
                      environment=env)

# submit the training script
run = exp.submit(config=src)


# Add name and tags for tracking
run.add_properties({'name': args.model_name})

# print run id to the stdout
print(run.id)

# save the run.
file = 'runids.csv'

# create header row if the file does not exist
if not os.path.exists(file):
    with open(file, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(['runid', 'model_name', 'data_subset', 'start_time'])

now = datetime.now()
# save info about this run
with open(file, 'a') as f:
    writer = csv.writer(f)
    writer.writerow([run.id, args.model_name, args.data_subset, now.strftime("%m/%d/%Y, %H:%M:%S")])
