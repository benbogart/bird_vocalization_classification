from azureml.tensorboard import Tensorboard
from azureml.core.run import Run
from azureml.core import Workspace, Experiment

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--runids', dest='runids',
                        nargs='+',
                        default=None,
                        help='runids to create')

    return parser.parse_args()

args = parse_args()

print(args)

if args.runids:
    # get workspace
    ws = Workspace.from_config()

    # set the expiriment
    experiment_name = 'test'
    exp = Experiment(workspace=ws, name=experiment_name)

    runs=[]
    for idx in args.runids:
        run = Run(exp, idx)
        runs.append(run)
    tb = Tensorboard(runs)
    tb.start()

    ## Wait for input to stop tensorboard.
    print('Enter to stop tensorboard')
    input()
    tb.stop()
