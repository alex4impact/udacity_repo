import pytest
from churn_library import import_data

@pytest.fixture(scope="module")
def path():
	return "../mlops_project_1_production_ready_code_for_ml/data/bank_data.csv"

# Creating a Dataframe object 'pytest.df' in Namespace
# def pytest_configure():
#     pytest.df = import_data(path)
