# outputs a list of directories that are incomplete

import os
import pickle

# data directories
seed_offsets = {'K_2':30000, 'K_5':20000, 'K_10':10000}
condition_keys = {'Aggregated':'aggregated','Compressed':'compressed', 'Unaggregated':'uaggregated'}
condition_offs = {'Aggregated':0,'Compressed':400,'Unaggregated':800}
save_extentions = {'K_2':'k2', 'K_5':'k5', 'K_10':'k10'}

# data features being saved
data_keys = ['testing_performance' ,'testing_complexity' ,'training_performance' ,'training_complexity' ,'task_id' ,'condition' ,'seed']

# tasks and replicates
classification_tasks = [146818,359954,359955,190146,168757,359956,359958,359959,2073,359960,168784,359962]
openml_tasks = classification_tasks
reps = 30
data_dir = f'{os.getenv("HOME")}/Desktop/Repositories/lexidate-cv-analysis/Results/'

def count_and_list_subdirectories(directory: str) -> int:
    """
    Count the number of subdirectories within a given directory,
    excluding the current (.) and root (..) directories. Also prints the subdirectories.

    Parameters:
    directory (str): The path of the directory.

    Returns:
    int: The number of subdirectories.
    """
    if not os.path.isdir(directory):
        raise ValueError(f"The provided path '{directory}' is not a valid directory.")

    subdirectories = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]

    failure_dirs = []

    for dir in subdirectories:
        if check_data_dir(f'{directory}{dir}') == False:
            failure_dirs.append(dir)

    return failure_dirs

# check if data was successfully collected
def check_data_dir(dir):
    # check if the directory exists
    if os.path.isdir(dir) == False:
        return False

    # check if the data was collected
    if os.path.isfile(f'{dir}/results.pkl') == False:
        return False

    # open the pkl file
    results = pickle.load(open(f'{dir}/results.pkl', 'rb'))

    # make sure results is of type dict
    if type(results) != dict:
        return False

    return True

# generate a list of directories to pull data from
def go_though_all_dirs():
    # collect all failed directories
    unfinished_dirs = []

    # go through each selection scheme, key, and task
    for k_cv, k_offset in seed_offsets.items():
        for condition, file_con in condition_keys.items():

            # current directory
            curr = f'/{k_cv}/{condition}/'
            failures_dirs = count_and_list_subdirectories(data_dir + curr)
            print(f'{curr}: {(reps * len(openml_tasks)) - len(failures_dirs)}/{reps * len(openml_tasks)}')

            # combine the failed directories
            unfinished_dirs.extend(failures_dirs)

    return unfinished_dirs

def main():
    failed_dirs = go_though_all_dirs()

    if len(failed_dirs) == 0:
        print('All directories are complete!')
        return

    # print out the directories that are incomplete
    print('Failed Directories:')
    for dir in failed_dirs:
        print(dir)

if __name__ == "__main__":
    main()