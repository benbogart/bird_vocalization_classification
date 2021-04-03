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
    parser.add_argument('--stop', dest='stop', action='store_const',
                        const=True, default=False,
                        help='Stop Tensorflow')

    return parser.parse_args()

args = parse_args()

print(args)

if args.stop:
#    tb = Tensorboard()
    Tensorboard([]).stop()

elif args.runids:
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
    print('Enter to stop tensorboard')
    input()
    tb.stop()
