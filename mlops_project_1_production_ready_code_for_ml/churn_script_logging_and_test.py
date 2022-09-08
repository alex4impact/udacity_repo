import os
import logging
import pytest
# from churn_library import import_data, perform_eda
import churn_library as cl

# @pytest.fixture
# def logger():
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
		request.config.cache.set('cache_df', df.to_json())
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


# def test_eda(path):
# 	'''
# 	test perform eda function
# 	'''
# 	try:
# 		 perform_eda(import_data(path))


# def test_encoder_helper(encoder_helper):
# 	'''
# 	test encoder helper
# 	'''

# 	df = encoder_helper(df, category_lst, response)

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
	# path = "./data/bank_data.csv"
	test_import(path, request)








