import pytest
import os
from typing import Generator, Any, Final


# See https://stackoverflow.com/questions/44677426/can-i-pass-arguments-to-pytest-fixtures
@pytest.fixture(scope="function")
def cleanup_files() -> Any:
    def _delete_file(*fpath) -> None:
        for f in fpath:
            os.remove(f)

    return _delete_file
