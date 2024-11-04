import argparse
import utils as utils

def main():
    # read in arguements
    parser = argparse.ArgumentParser()
    # split proportion we are using
    parser.add_argument("-cv_type", required=True, nargs='?')
    # what openml task are we using
    parser.add_argument("-task_id", required=True, nargs='?')
    # number of threads to use during estimator evalutation
    parser.add_argument("-n_jobs",  required=True, nargs='?')
    # where to save the results/models
    parser.add_argument("-savepath", required=True, nargs='?')
    # seed offset
    parser.add_argument("-seed", required=True, nargs='?')

    args = parser.parse_args()
    cv_type = str(args.cv_type)
    print('Split:', cv_type)
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
    classification_tasks = [146818,359958,359963,190411,168350]

    assert task_id in classification_tasks, 'Task ID not in list of tasks'

    # execute task
    utils.execute_experiment(cv_type,task_id,n_jobs,savepath,seed)

if __name__ == '__main__':
    main()
    print('FINISHED')