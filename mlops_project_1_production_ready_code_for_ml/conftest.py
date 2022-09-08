import pytest
from constants import DF_PATH


@pytest.fixture(scope="module")
def path():
    return DF_PATH
    