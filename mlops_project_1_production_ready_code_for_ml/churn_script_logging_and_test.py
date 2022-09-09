import os
import sys
import logging
import pandas as pd
import pandas.api.types as ptypes
import churn_library as cl
from constants import PLOTS, PLOTS_PATH, CATEGORY_LIST, KEEP_COLS, RESPONSE


logging.basicConfig(
	filename='../mlops_project_1_production_ready_code_for_ml/logs/churn_library.log',
	level=logging.INFO,
	filemode='w',
	format='%(name)s - %(levelname)s - %(message)s - %(asctime)s')


def test_import(path, request):
	'''
	test data import - this example is completed for you to assist with the other test functions
	'''
	try:
		df = cl.import_data(path)
		request.config.cache.set('data/df', df.to_dict())
		logging.info("Testing import_data: SUCCESS")
	except FileNotFoundError as err:
		logging.error("Testing import_eda: The file wasn't found")
		raise err

	try:
		assert df.shape[0] > 0
		assert df.shape[1] > 0
	except AssertionError as err:
		logging.error("Testing import_data: The file doesn't appear to have rows and columns")
		raise err


def test_eda(request):
	'''
	test perform eda function
	'''
	try:
		df = pd.DataFrame(request.config.cache.get('data/df', None))
		assert df.shape[0] > 0
		assert df.shape[1] > 0
		logging.info("Cached df loaded: SUCCESS")
	except AssertionError as err:
		logging.error("Testing cached df loaded on test_eda function:\
			The file doesn't appear to have rows and columns")
		raise err
	try:
		cl.perform_eda(df, PLOTS, PLOTS_PATH)
		assert len([entry for entry in os.listdir(PLOTS_PATH) if os.path.isfile(os.path.join(PLOTS_PATH, entry))]) == len(PLOTS)
		logging.info("Number of plot files is correct: SUCCESS")
		for plot in PLOTS:
			if str.lower(plot) != 'heatmap':
				assert os.path.isfile(os.path.join(PLOTS_PATH, f'{str.lower(plot)}_distribution.png'))
				logging.info(f"{str.lower(plot)}_distribution.png exists: SUCCESS")
			else:
				assert os.path.isfile(os.path.join(PLOTS_PATH, f'{str.lower(plot)}.png'))
				logging.info(f"{str.lower(plot)}.png exists: SUCCESS")
	except AssertionError as err:
		logging.error("Testing plots saved on test_eda function:\
			The number of plots is incorrect or one or more plots don't exist")
		raise err


def test_encoder_helper(request):
	'''
	test encoder helper
	'''
	try:
		df = pd.DataFrame(request.config.cache.get('data/df', None))
		assert df.shape[0] > 0
		assert df.shape[1] > 0
		logging.info("Cached df loaded: SUCCESS")
	except AssertionError as err:
		logging.error("Testing cached df loaded on test_encoder_helper function:\
			The file doesn't appear to have rows and columns")
		raise err
	try:
		alt_df = df.copy()
		new_df = cl.encoder_helper(alt_df, CATEGORY_LIST, RESPONSE)
		assert len(new_df.columns) == len(df.columns) + len(CATEGORY_LIST)
		assert new_df.shape[0] == df.shape[0]
		assert all(ptypes.is_numeric_dtype(new_df[f'{col}_{RESPONSE}']) for col in CATEGORY_LIST)
		logging.info("Categorical columns encoded by encoder_helper: SUCCESS")
	except AssertionError as err:
		logging.error("Testing encoding of categorical columns on test_encoder_helper function:\
			Categorical columns haven't been encoded correctly or encoded df has different number of records")
		raise err


def test_perform_feature_engineering(request):
	'''
	test perform_feature_engineering
	'''
    
	try:
		df = pd.DataFrame(request.config.cache.get('data/df', None))
		assert df.shape[0] > 0
		assert df.shape[1] > 0
		logging.info("Cached df loaded: SUCCESS")
		X_train, X_test, y_train, y_test = cl.perform_feature_engineering(df, CATEGORY_LIST, KEEP_COLS, RESPONSE)
		assert list(X_train.columns) == KEEP_COLS and list(X_test.columns) == KEEP_COLS
		assert df.shape[0] > X_train.shape[0] > X_test.shape[0] > 0
		assert df.shape[1] > X_train.shape[1] == X_test.shape[1] > 0
		assert all(ptypes.is_numeric_dtype(X_train[item]) for item in KEEP_COLS)
		assert all(ptypes.is_numeric_dtype(X_test[item]) for item in KEEP_COLS)
		request.config.cache.set('data/X_train', X_train.to_dict())
		request.config.cache.set('data/X_test', X_test.to_dict())
		logging.info("X_train and X_test split done correctly: SUCCESS")
		assert df.shape[0] > y_train.shape[0] > y_test.shape[0] > 0
		assert len(y_train.shape) == len(y_test.shape) > 0
		assert all(ptypes.is_numeric_dtype(item) for item in [y_train, y_test])
		request.config.cache.set('data/y_train', y_train.to_dict())
		request.config.cache.set('data/y_test', y_test.to_dict())
		logging.info("y_train and y_test split done correctly: SUCCESS")
	except AssertionError as err:
		logging.error("Testing feature engineering function:\
			df split into X_train, X_test, y_train, y_test is incorrect or types are not correct")
		raise err


def test_train_models(request):
	'''
	test train_models
	'''
	train_models(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
	test_import(path, request)
	test_eda(request)
	test_encoder_helper(request)
	test_perform_feature_engineering(request)








