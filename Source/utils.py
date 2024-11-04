import openml
import tpot2
from tpot2.search_spaces.pipelines import ChoicePipeline, SequentialPipeline
import sklearn.metrics
import sklearn
import traceback
import dill as pickle
import os
import time
import numpy as np
import sklearn.model_selection
from functools import partial
from estimator_node_gradual import EstimatorNodeGradual

# lexicase selection with ignoring the complexity column
def lexicase_selection_no_complexity(scores, k, rng=None, n_parents=1,):
    """
    Select the best individual according to Lexicase Selection, *k* times.
    The returned list contains the indices of the chosen *individuals*.
    :param scores: The score matrix, where rows the individulas and the columns are the corresponds to scores on different objectives.
    :returns: A list of indices of selected individuals.
    """
    rng = np.random.default_rng(rng)
    chosen =[]

    for _ in range(k*n_parents):
        candidates = list(range(len(scores)))
        cases = list(range(len(scores[0]) - 1)) # ignore the last column which is complexity
        rng.shuffle(cases)

        while len(cases) > 0 and len(candidates) > 1:
            best_val_for_case = max(scores[candidates,cases[0]])
            candidates = [x for x in candidates if scores[x, cases[0]] == best_val_for_case]
            cases.pop(0)
        chosen.append(rng.choice(candidates))

    return np.reshape(chosen, (k, n_parents))

# generate traditional cross validation scores for tournament selection
def compressed_selection_objectives(est,X,y,cv):
    # hold all the scores
    scores = []
    complexity = []

    for train_index, test_index in cv.split(X, y):
        # make a copy of the estimator
        this_fold_pipeline = sklearn.base.clone(est)

        # get data split
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # fit model
        this_fold_pipeline.fit(X_train, y_train)

        # append complexity score
        complexity += [np.int64(tpot2.objectives.complexity_scorer(this_fold_pipeline,0,0))]
        # append accuracy score
        scores += [np.mean(np.int64(this_fold_pipeline.predict(X_test) == y_test), dtype=np.float64)]

        del this_fold_pipeline
        del X_train
        del X_test
        del y_train
        del y_test

    # make sure we have the right number of scores
    assert len(scores) == 10
    assert len(complexity) == 10
    return np.mean(scores, dtype=np.float64), np.mean(complexity, dtype=np.float64)

# generate individual fold cross validation scores for lexicase selection
def aggregated_selection_objectives(est,X,y,cv):
    # hold all the scores
    scores = []
    complexity = []

    for train_index, test_index in cv.split(X, y):
        # make a copy of the estimator
        this_fold_pipeline = sklearn.base.clone(est)

        # get data split
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # fit model
        this_fold_pipeline.fit(X_train, y_train)

        # append complexity score
        complexity += [np.int64(tpot2.objectives.complexity_scorer(this_fold_pipeline,0,0))]
        # append accuracy score
        scores += [np.mean(np.int64(this_fold_pipeline.predict(X_test) == y_test), dtype=np.float64)]

        del this_fold_pipeline
        del X_train
        del X_test
        del y_train
        del y_test

    # make sure we have the right number of scores
    assert len(scores) == 10
    assert len(complexity) == 10
    return scores + [np.mean(complexity, dtype=np.float64)]

# generate individual fold cross validation scores for lexicase selection
def unaggregated_selection_objectives(est,X,y,cv):
    # hold all the scores
    scores = []
    complexity = []

    for train_index, test_index in cv.split(X, y):
        # make a copy of the estimator
        this_fold_pipeline = sklearn.base.clone(est)

        # get data split
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # fit model
        this_fold_pipeline.fit(X_train, y_train)

        # append complexity score
        complexity += [np.int64(tpot2.objectives.complexity_scorer(this_fold_pipeline,0,0))]
        # append accuracy score
        scores += list(np.int64(this_fold_pipeline.predict(X_test) == y_test))

        del this_fold_pipeline
        del X_train
        del X_test
        del y_train
        del y_test

    # make sure we have the right number of scores
    assert len(scores) == X.shape[0]
    assert len(complexity) == 10
    return scores + [np.mean(complexity, dtype=np.float64)]

# pipeline search space: selector(required) -> transformer(optional) -> regressor/classifier(required)
def get_pipeline_space(seed):
    return tpot2.search_spaces.pipelines.SequentialPipeline([
        tpot2.config.get_search_space("selectors_classification", random_state=seed, base_node=EstimatorNodeGradual),
        tpot2.config.get_search_space(["transformers","Passthrough"], random_state=seed, base_node=EstimatorNodeGradual),
        tpot2.config.get_search_space("classifiers", random_state=seed, base_node=EstimatorNodeGradual)])

# get selection scheme
def get_selection_scheme(cv_type):
    if cv_type == 'compressed':
        return tpot2.selectors.tournament_selection
    elif cv_type == 'aggregated':
        return lexicase_selection_no_complexity
    elif cv_type == 'unaggregated':
        return lexicase_selection_no_complexity
    else:
        raise ValueError(f"Unknown selection scheme: {cv_type}")

# get estimator parameters depending on the selection scheme
def get_estimator_params(n_jobs,
                         cv_type,
                         X_train,
                         y_train,
                         seed):
    # print data shapes
    print('X_train:',X_train.shape,'|','y_train:',y_train.shape)

    # generate cv split
    cv = sklearn.model_selection.StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

    # get selection objective functions
    if cv_type == 'compressed':
        # create selection objective functions
        objective_scorer = partial(compressed_selection_objectives,X=X_train,y=y_train,cv=cv)
        objective_scorer.__name__ = 'compressed-complexity'
        # cv_score + complexity
        objective_names = ['cv'] + ['complexity']
        objective_weights = [1.0] + [-1.0]
    elif cv_type == 'aggregated':
        # create selection objective functions
        objective_scorer = partial(aggregated_selection_objectives,X=X_train,y=y_train,cv=cv)
        objective_scorer.__name__ = 'aggregated-complexity'
        # accuracy_per_fold + complexity
        objective_names = ['fold_'+str(i) for i in range(10)] + ['complexity']
        objective_weights = [1.0 for _ in range(10)] + [-1.0]
    elif cv_type == 'unaggregated':
        # create selection objective functions
        objective_scorer = partial(unaggregated_selection_objectives,X=X_train,y=y_train,cv=cv)
        objective_scorer.__name__ = 'unaggregated-complexity'
        # accuracy_per_sample + complexity
        objective_names = ['sample_'+str(i) for i in range(X_train.shape[0])] + ['complexity']
        objective_weights = [1.0 for _ in range(X_train.shape[0])] + [-1.0]
    else:
        raise ValueError(f"Unknown selection scheme: {cv_type}")

    return cv, {
        # evaluation criteria
        'scorers': [],
        'scorers_weights':[],
        'cv': sklearn.model_selection.StratifiedKFold(n_splits=10, shuffle=True, random_state=seed), # not used
        'other_objective_functions': [objective_scorer],
        'other_objective_functions_weights': objective_weights,
        'objective_function_names': objective_names,

        # evolutionary algorithm params
        'population_size' : 10,
        'generations' : 2,
        'n_jobs':n_jobs,
        'survival_selector' :None,
        'parent_selector': get_selection_scheme(cv_type),
        'random_state': seed,

        # offspring variation params
        'mutate_probability': 0.7,
        'crossover_probability': 0.0,
        'crossover_then_mutate_probability': 0.3,
        'mutate_then_crossover_probability': 0.0,

        # estimator params
        'memory_limit':0,
        'preprocessing':False,
        'classification' : True,
        'verbose':1,
        'max_eval_time_mins':10, # 10 min time limit
        'max_time_mins': float("inf"), # run until generations are done

        # pipeline search space
        'search_space': get_pipeline_space(seed)
        }

# get test scores
def score(est, X, y, X_train, y_train):
    # train evovled pipeline on the training data
    est.fit(X_train, y_train)
    performance = np.float64(sklearn.metrics.get_scorer("accuracy")(est, X, y))
    return {'testing_performance': performance, 'testing_complexity': np.int64(tpot2.objectives.complexity_scorer(est,0,0))}

#https://github.com/automl/ASKL2.0_experiments/blob/84a9c0b3af8f7ac6e2a003d4dea5e6dce97d4315/experiment_scripts/utils.py
def load_task(task_id, classification, preprocess=True):

    cached_data_path = f"data/{task_id}_{preprocess}.pkl"
    if os.path.exists(cached_data_path):
        d = pickle.load(open(cached_data_path, "rb"))
        X_train, y_train, X_test, y_test = d['X_train'], d['y_train'], d['X_test'], d['y_test']
    else:
        task = openml.tasks.get_task(task_id)

        X, y = task.get_X_and_y(dataset_format="dataframe")
        train_indices, test_indices = task.get_train_test_split_indices()
        X_train = X.iloc[train_indices]
        y_train = y.iloc[train_indices]
        X_test = X.iloc[test_indices]
        y_test = y.iloc[test_indices]

        if preprocess:
            preprocessing_pipeline = sklearn.pipeline.make_pipeline(tpot2.builtin_modules.ColumnSimpleImputer("categorical", strategy='most_frequent'), tpot2.builtin_modules.ColumnSimpleImputer("numeric", strategy='mean'), tpot2.builtin_modules.ColumnOneHotEncoder("categorical", min_frequency=0.001, handle_unknown="ignore"))
            X_train = preprocessing_pipeline.fit_transform(X_train)
            X_test = preprocessing_pipeline.transform(X_test)

            # needed this to LabelEncode the target variable if it is a classification task only
            if classification:
                le = sklearn.preprocessing.LabelEncoder()
                y_train = le.fit_transform(y_train)
                y_test = le.transform(y_test)

            X_train = X_train.to_numpy()
            X_test = X_test.to_numpy()

            if task_id == 168795: #this task does not have enough instances of two classes for 10 fold CV. This function samples the data to make sure we have at least 10 instances of each class
                indices = [28535, 28535, 24187, 18736,  2781]
                y_train = np.append(y_train, y_train[indices])
                X_train = np.append(X_train, X_train[indices], axis=0)

            d = {"X_train": X_train, "y_train": y_train, "X_test": X_test, "y_test": y_test}
            if not os.path.exists("data"):
                os.makedirs("data")
            with open(cached_data_path, "wb") as f:
                pickle.dump(d, f)

    return X_train, y_train, X_test, y_test

# get the best pipeline from tpot2 depending on the selection scheme
def get_best_pipeline_results(est, cv, cv_type, seed, X, y):
    # update this subset of data to get the best performer
    sub = est.evaluated_individuals

    # if unaggregated, must get the average of all scores for each fold
    if cv_type == 'unaggregated':
        # remove rows with missing values
        sub = sub.dropna(subset=['sample_0'])

        # get scores for each fold
        i = 0
        for _, test_index in cv.split(X, y):
            # create test sample in each fold
            fold_names = ['sample_'+str(id) for id in test_index]
            # average all scores for a fold and add it as a new column
            sub['fold_'+str(i)] = sub[fold_names].mean(axis=1)
            i += 1
        assert i == 10

    # if aggregated, must get the average of all scores for each fold into a single score
    if cv_type == 'aggregated' or cv_type == 'unaggregated':
        # remove rows with missing values
        sub = sub.dropna(subset=['fold_0'])
        # average all scores for a fold and add it as a new column
        sub['cv'] = sub[[f'fold_{i}' for i in range(10)]].mean(axis=1)

    # drop all NA values for 'cv' column
    sub = sub.dropna(subset=['cv'])

    # get best performers based on classification accuracy maximization
    best_performers = sub[sub['cv'] == sub['cv'].max()]

    # filter by the smallest complexity
    best_performers = best_performers[best_performers['complexity'] == best_performers['complexity'].min()]

    # get best performer performance and cast to numpy float32
    best_performer =  best_performers.sample(1, random_state=seed)

    # return performance, complexity, and individual
    return np.float64(best_performer['cv'].values[0]), \
                np.int64(best_performer['complexity'].values[0]), \
                best_performer['Individual'].values[0].export_pipeline()

# execute task with tpot2
def execute_experiment(cv_type,task_id,n_jobs,savepath,seed):
    # generate directory to save results
    save_folder = f"{savepath}/{seed}-{task_id}"
    if os.path.exists(save_folder):
        print('FOLDER ALREADY EXISTS:', save_folder)
        return

    # run experiment
    try:
        print("LOADING DATA")
        X_train, y_train, X_test, y_test = load_task(task_id, preprocess=True, classification=True)

        # get estimator parameters
        cv, est_params = get_estimator_params(n_jobs=n_jobs,cv_type=cv_type,X_train=X_train,y_train=y_train,seed=seed)
        est = tpot2.TPOTEstimator(**est_params)

        start = time.time()
        print("ESTIMATOR FITTING")
        est.fit(X_train, y_train) # x_train, y_train are not used at all by the estimator
        duration = time.time() - start
        print("ESTIMATOR FITTING COMPLETE:", duration / 60 / 60, 'hours')

        # get best performer performance and cast to numpy float32
        train_performance, complexity, pipeline = get_best_pipeline_results(est, cv, cv_type, seed, X_train, y_train)

        # get test scores and save results
        results = score(pipeline, X_test, y_test, X_train=X_train, y_train=y_train)
        results['training_performance'] = train_performance
        results['training_complexity'] = complexity
        results["task_id"] = task_id
        results["cv_type"] = cv_type
        results["seed"] = seed

        print('RESULTS:', results)

        print('CREATING FOLDER:', save_folder)
        os.makedirs(save_folder)

        print('SAVING:SCORES.PKL')
        with open(f"{save_folder}/results.pkl", "wb") as f:
            pickle.dump(results, f)
        return

    except Exception as e:
        trace =  traceback.format_exc()
        pipeline_failure_dict = {"task_id": task_id, "cv_type": cv_type, "seed": seed, "error": str(e), "trace": trace}
        print("failed on ")
        print(save_folder)
        print(e)
        print(trace)

        with open(f"{save_folder}/failed.pkl", "wb") as f:
            pickle.dump(pipeline_failure_dict, f)

    return