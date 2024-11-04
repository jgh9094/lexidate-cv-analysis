# Description: This script is used to run the hold out experiment for the preliminary tasks.≠≠≠≠≠≠≠≠≠≠

from utils import load_task

def main():
    # Classification tasks from the 'AutoML Benchmark All Classification' suite
    # Suite is used within 'AMLB: an AutoML Benchmark' paper
    # https://github.com/openml/automlbenchmark
    # https://www.jmlr.org/papers/volume25/22-0493/22-0493.pdf
    # https://www.openml.org/search?type=benchmark&study_type=task&sort=tasks_included&id=271

    # classification tasks:
    # 100 < rows < 2000
    # columns < 10000
    classification_tasks = [359953,146818,359954,359955,190146,168757,359956,
                            359957,359958,359959,2073,10090,359960,168784,359961,
                            359962]

    for task_id in classification_tasks:
        print('processing task:', task_id)
        load_task(task_id, 'classification')
    return

if __name__ == '__main__':
    main()
    print('FINISHED')