# outputs a list of directories that are incomplete

import os
import pickle
import pandas as pd

# data directories
condition_keys = {'Aggregated':'aggregated','Compressed':'compressed', 'Unaggregated':'uaggregated'}
save_extentions = {'K_2':'k2', 'K_5':'k5', 'K_10':'k10'}

keys = ['testing_performance', 'testing_complexity', 'training_performance', 'training_complexity', 'task_id', 'validation', 'seed']

k2_data = {'testing_performance': [],
           'testing_complexity': [],
           'training_performance': [],
           'training_complexity': [],
           'task_id': [],
           'validation': [],
           'seed': []}

k5_data = {'testing_performance': [],
           'testing_complexity': [],
           'training_performance': [],
           'training_complexity': [],
           'task_id': [],
           'validation': [],
           'seed': []}

k10_data = {'testing_performance': [],
           'testing_complexity': [],
           'training_performance': [],
           'training_complexity': [],
           'task_id': [],
           'validation': [],
           'seed': []}

# tasks and replicates
classification_tasks = [146818,359954,359955,190146,168757,359956,359958,359959,2073,359960,168784,359962]
openml_tasks = classification_tasks
reps = 30
data_dir = f'{os.getenv("HOME")}/Desktop/Repositories/lexidate-cv-analysis/Results/'
save_dir = f'{os.getenv("HOME")}/Desktop/Repositories/lexidate-cv-analysis/'

def list_subdirectories(directory: str, condition: str) -> int:
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

    for dir in subdirectories:
        # res =  extract_data(f'{directory}{dir}')
        results = pickle.load(open(f'{directory}{dir}/results.pkl', 'rb'))

        for key in keys:
            if condition == 'K_2':
                k2_data[key].append(results[key])
            elif condition == 'K_5':
                k5_data[key].append(results[key])
            elif condition == 'K_10':
                k10_data[key].append(results[key])
            else:
                exit('Invalid condition key')
    return

# generate a list of directories to pull data from
def go_though_all_dirs():
    # collect all failed directories

    # go through each selection scheme, key, and task
    for k_cv, k_save in save_extentions.items():
        for condition, file_con in condition_keys.items():
            list_subdirectories(data_dir + f'/{k_cv}/{condition}/', k_cv)

    return

def main():
    go_though_all_dirs()

    # save each dictionary into a csv file with pandas
    k2_df = pd.DataFrame(k2_data)
    k5_df = pd.DataFrame(k5_data)
    k10_df = pd.DataFrame(k10_data)

    k2_df.to_csv(f'{save_dir}/k2_data.csv', index=False)
    k5_df.to_csv(f'{save_dir}/k5_data.csv', index=False)
    k10_df.to_csv(f'{save_dir}/k10_data.csv', index=False)

if __name__ == "__main__":
    main()