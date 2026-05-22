.PHONY: install format lint typecheck test smoke clean

PYTHON ?= python

install:
	$(PYTHON) -m pip install -e ".[dev]"

format:
	$(PYTHON) -m ruff format .
	$(PYTHON) -m ruff check --fix .

lint:
	$(PYTHON) -m ruff check .

typecheck:
	$(PYTHON) -m mypy chronoslob

test:
	$(PYTHON) -m pytest

smoke:
	$(PYTHON) -c "import chronoslob; print(chronoslob.__version__)"
	$(PYTHON) -m chronoslob.cli version
	$(PYTHON) -m chronoslob.cli doctor

clean:
	$(PYTHON) -c "from pathlib import Path; import shutil; [shutil.rmtree(p, ignore_errors=True) for p in Path('.').rglob('__pycache__')]; [shutil.rmtree(p, ignore_errors=True) for p in [Path('.pytest_cache'), Path('.mypy_cache'), Path('.ruff_cache'), Path('htmlcov')]]"
