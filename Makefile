.PHONY: tests
tests:
	PY_MAJOR_VERSION=py`python -c 'import sys; print(sys.version_info[0])'` pytest -v --cov=nmfx tests
	flake8 nmfx
