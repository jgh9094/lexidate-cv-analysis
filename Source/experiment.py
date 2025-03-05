# to run: clear; python experiment.py -validation compressed -k 0 -task_id 146818 -n_jobs 10 -savepath -seed 0

import argparse
import utils as utils
import os

def main():
    # read in arguements
    parser = argparse.ArgumentParser()
    # split proportion we are using
    parser.add_argument("-validation", required=True, nargs='?')
    # k value for k-fold cross validation
    parser.add_argument("-k", required=False, nargs='?')
    # what openml task are we using
    parser.add_argument("-task_id", required=True, nargs='?')
    # number of threads to use during estimator evalutation
    parser.add_argument("-n_jobs",  required=True, nargs='?')
    # where to save the results/models
    parser.add_argument("-savepath", required=True, nargs='?')
    # seed offset
    parser.add_argument("-seed", required=True, nargs='?')

    args = parser.parse_args()
    validation = str(args.validation)
    print('Validation:', validation)
    k = int(args.k)
    print('K:', k)
    task_id = int(args.task_id)
    print('Task ID:', task_id)
    n_jobs = int(args.n_jobs)
    print('Number of Jobs:', n_jobs)
    savepath = str(args.savepath)
    print('Save Path:', savepath)
    seed = int(args.seed)
    print('Seed:', seed)

    # Classification tasks from the 'AutoML Benchmark All Classification' suite
    # Suite is used within 'AMLB: an AutoML Benchmark' paper
    # https://github.com/openml/automlbenchmark
    # https://www.jmlr.org/papers/volume25/22-0493/22-0493.pdf
    # https://www.openml.org/search?type=benchmark&study_type=task&sort=tasks_included&id=271

    # classification tasks:
    # rows < 5000
    # columns < 500
    classification_tasks = [146818,359954,359955,190146,168757,359956,
                            359958,359959,2073,359960,168784,359962]

    assert task_id in classification_tasks, 'Task ID not in list of tasks'

    # execute task
    utils.execute_experiment(validation,task_id,n_jobs,savepath,seed,k)

if __name__ == '__main__':
    main()
    print('FINISHED')