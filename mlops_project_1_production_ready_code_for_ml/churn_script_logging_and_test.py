import os
import sys
import logging
import pandas as pd
# import pytest
# from churn_library import import_data, perform_eda
import churn_library as cl
from constants import PLOTS, PLOTS_PATH, CATEGORY_LIST, RESPONSE

# # Parent Folder 
# sys.path.append(
#     os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
# )


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
		cl.perform_eda(df)
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
		logging.info("Categorical columns encoded by encoder_helper: SUCCESS")
	except AssertionError as err:
		logging.error("Testing encoding of categorical columns on test_encoder_helper function:\
			Categorical columns haven't been encoded or new df has different number of records")
		raise err







# def test_perform_feature_engineering(perform_feature_engineering):
# 	'''
# 	test perform_feature_engineering
# 	'''
#     response = 'Churn'
#     X_train, X_test, y_train, y_test = perform_feature_engineering(df, response)


# def test_train_models(train_models):
# 	'''
# 	test train_models
# 	'''
# 	train_models(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
	test_import(path, request)








