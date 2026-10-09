import os
import sys

import pytest
from jaxtyping import install_import_hook

# Check every jaxtyping shape annotation at runtime while tests run. Must be installed before okur is imported.
_hook = install_import_hook("okur", "beartype.beartype")

_status = 0


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    global _status  # noqa: PLW0603
    _status = int(exitstatus)


def pytest_unconfigure(config: pytest.Config) -> None:
    """ONNX Runtime and MLX in one process abort in native teardown (libc++ recursive_mutex) after all tests have
    finished and been reported. Exit with the real status before that teardown runs."""
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(_status)
