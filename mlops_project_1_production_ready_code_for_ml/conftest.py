import pytest
import os 
import sys


# Parent Folder 
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
)


@pytest.fixture(scope="module")
def path():
    FILE_PATH = "bank_data.csv"
    return os.path.dirname(os.path.realpath(__file__)) + "/data/" + FILE_PATH
    # "../mlops_project_1_production_ready_code_for_ml/data/"