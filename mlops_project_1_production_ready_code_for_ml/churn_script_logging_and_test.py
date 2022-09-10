"""Testing module to test the churn library functions

Raises:
	FileNotFoundError: when testing import_eda:
		The file wasn't found.
	AssertionError: when testing import_data:
		The file doesn't appear to have rows and columns.
	AssertionError: when testing cached df loaded on test_eda function:
		The cached df doesn't appear to have rows and columns.
	AssertionError: when testing plots saved on test_eda function:
		The number of plots is incorrect or one or more plots don't exist.
	AssertionError: when testing encoding of categorical columns on test_encoder_helper function:\
		Categorical columns haven't been encoded correctly or encoded df has \
		different number of records
	AssertionError: when testing feature engineering function:
		df split into X_train, X_test, y_train, y_test is incorrect or types are not correct
	AssertionError: when testing models training function:
		Model training or output results are not correct.

author: Alex Carvalho
date: September 2022
"""

import os
import imghdr
import logging
import pandas as pd
import pandas.api.types as ptypes
import churn_library as cl
from constants import DF_PATH, PLOTS, PLOTS_PATH, CATEGORY_LIST, KEEP_COLS,\
    RESPONSE, MODELS_PATH, MODELS, RESULTS_PATH, PARAM_GRID


logging.basicConfig(
    filename='../mlops_project_1_production_ready_code_for_ml/logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s - %(asctime)s')


def test_import(request):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        data = cl.import_data(DF_PATH)
        request.config.cache.set('data/data', data.to_dict())
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert data.shape[0] > 0
        assert data.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda(request):
    '''
    test perform eda function
    '''
    try:
        data = pd.DataFrame(request.config.cache.get('data/data', None))
        assert data.shape[0] > 0
        assert data.shape[1] > 0
        logging.info("Cached df loaded: SUCCESS")
    except AssertionError as err:
        logging.error("Testing cached df loaded on test_eda function:\
			The cached df doesn't appear to have rows and columns")
        raise err
    try:
        cl.perform_eda(data, PLOTS, PLOTS_PATH)
        assert len([entry for entry in os.listdir(PLOTS_PATH) if os.path.isfile(
            os.path.join(PLOTS_PATH, entry))]) == len(PLOTS)
        logging.info("Number of plot files is correct: SUCCESS")
        for plot in PLOTS:
            if str.lower(plot) != 'heatmap':
                assert os.path.isfile(
                    os.path.join(
                        PLOTS_PATH,
                        f'{str.lower(plot)}_distribution.png'))
                logging.info(
                    "%s_distribution.png exists: SUCCESS", str.lower(plot))
            else:
                assert os.path.isfile(
                    os.path.join(
                        PLOTS_PATH,
                        f'{str.lower(plot)}.png'))
                logging.info("%s.png exists: SUCCESS", str.lower(plot))
    except AssertionError as err:
        logging.error("Testing plots saved on test_eda function:\
			The number of plots is incorrect or one or more plots don't exist")
        raise err


def test_encoder_helper(request):
    '''
    test encoder helper
    '''
    try:
        data = pd.DataFrame(request.config.cache.get('data/data', None))
        assert data.shape[0] > 0
        assert data.shape[1] > 0
        logging.info("Cached df loaded: SUCCESS")
        alt_data = data.copy()
        new_data = cl.encoder_helper(alt_data, CATEGORY_LIST, RESPONSE)
        assert len(new_data.columns) == len(data.columns) + len(CATEGORY_LIST)
        assert new_data.shape[0] == data.shape[0]
        assert all(
            ptypes.is_numeric_dtype(
                new_data[f'{col}_{RESPONSE}']) for col in CATEGORY_LIST)
        logging.info("Categorical columns encoded by encoder_helper: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing encoding of categorical columns on test_encoder_helper function:\
				Categorical columns haven't been encoded correctly or encoded df has \
				different number of records")
        raise err


def test_perform_feature_engineering(request):
    '''
    test perform_feature_engineering
    '''

    try:
        data = pd.DataFrame(request.config.cache.get('data/data', None))
        assert data.shape[0] > 0
        assert data.shape[1] > 0
        logging.info("Cached df loaded: SUCCESS")
        X_train, X_test, y_train, y_test = cl.perform_feature_engineering(
            data, CATEGORY_LIST, KEEP_COLS, RESPONSE)
        assert list(
            X_train.columns) == KEEP_COLS and list(
            X_test.columns) == KEEP_COLS
        assert data.shape[0] > X_train.shape[0] > X_test.shape[0] > 0
        assert data.shape[1] > X_train.shape[1] == X_test.shape[1] == len(
            KEEP_COLS) > 1
        assert all(
            ptypes.is_numeric_dtype(
                X_train[item]) for item in KEEP_COLS)
        assert all(ptypes.is_numeric_dtype(X_test[item]) for item in KEEP_COLS)
        request.config.cache.set('data/X_train', X_train.to_dict())
        request.config.cache.set('data/X_test', X_test.to_dict())
        logging.info("X_train and X_test split done correctly: SUCCESS")
        assert data.shape[0] > y_train.shape[0] > y_test.shape[0] > 0
        assert len(y_train.shape) == len(y_test.shape) > 0
        assert all(ptypes.is_numeric_dtype(item) for item in [y_train, y_test])
        request.config.cache.set('data/y_train', y_train.to_dict())
        request.config.cache.set('data/y_test', y_test.to_dict())
        logging.info("y_train and y_test split done correctly: SUCCESS")
    except AssertionError as err:
        logging.error("Testing feature engineering function:\
			df split into X_train, X_test, y_train, y_test is incorrect\
			or types are not correct")
        raise err


def test_train_models(request):
    '''
    test train_models
    '''

    try:
        X_train = pd.DataFrame(request.config.cache.get('data/X_train', None))
        X_test = pd.DataFrame(request.config.cache.get('data/X_test', None))
        y_train = pd.Series(request.config.cache.get('data/y_train', None))
        y_test = pd.Series(request.config.cache.get('data/y_test', None))
        logging.info("Cached X_train, X_test, y_train, y_test loaded: SUCCESS")
        cl.train_models(X_train, X_test, y_train, y_test, PARAM_GRID, MODELS,
                        MODELS_PATH, RESULTS_PATH)
        logging.info("Trained model run completed: SUCCESS")
        assert len([entry for entry in os.listdir(MODELS_PATH) if os.path.isfile(
            os.path.join(MODELS_PATH, entry))]) == len(MODELS)
        logging.info("Number of model files is correct: SUCCESS")
        for model in MODELS:
            assert os.path.isfile(os.path.join(MODELS_PATH, model))
            logging.info("model file %s exists: SUCCESS", model)
        logging.info(
            "Checking if there are 5 different png files in the results folder...")
        assert len({entry for entry in os.listdir(RESULTS_PATH) if
                        os.path.isfile(os.path.join(RESULTS_PATH, entry)) and
                        imghdr.what(os.path.join(RESULTS_PATH, entry)) == 'png'}) == 5
        logging.info(
            "There are 5 different png files in the results folder: SUCCESS")
    except AssertionError as err:
        logging.error("Testing models training function:\
			Model training or output results are not correct")
        raise err
